#include <boost/mpi.hpp>
#include <boost/program_options.hpp>
#include <fmt/format.h>

#include "config.h"
#include "gridspec.h"
#include "logger.h"
#include "mpi.h"
#include "routines.h"

int main(int argc, char** argv) {
    boost::mpi::environment env(argc, argv, boost::mpi::threading::multiple);
    boost::mpi::communicator world;

    Config config;

    bool ok {false};  // Signal to continue after command line parsing

    if (world.rank() == 0) {
        namespace po = boost::program_options;

        po::options_description desc;
        desc.add_options()
            ("help,h", "Output this help text")
            ("config", po::value<std::string>(), "Path to .toml configuration file")
            ("makeconfig", "Print the full default .toml configuration and exit")
            (
                "mset",
                po::value<std::vector<std::string>>(&config.msets)->multitoken()
                                                                  ->zero_tokens()
                                                                  ->composing(),
                "Measurement set path"
            );

        po::positional_options_description pdesc;
        pdesc.add("mset", -1);

        po::command_line_parser parser {argc, argv};
        parser.options(desc).positional(pdesc);

        po::variables_map vm;

        auto printheader = [] (auto& out) {
            fmt::println(out, "Pigi: the parallel interferomentric GPU imager");
            fmt::println(
                out,
                "Usage: [--help] [--config config.toml] [--makeconfig] [mset, [mset...]]\n"
            );
        };

        try {
            po::store(parser.run(), vm);
            po::notify(vm);
        } catch (const po::error& e) {
            printheader(stderr);
            std::cerr << e.what() << std::endl;

            boost::mpi::broadcast(world, ok, 0);
            return -1;
        }

        if (vm.count("help")) {
            printheader(stdout);
            std::cout << desc << std::endl;

            boost::mpi::broadcast(world, ok, 0);
            return 0;
        }

        if (vm.count("makeconfig")) {
            std::cout << std::setw(160)
                      << toml::basic_value<toml::preserve_comments>(config)
                      << std::endl;
            boost::mpi::broadcast(world, ok, 0);
            return 0;
        }

        if (vm.count("config")) {
            auto v = toml::expect<Config>(
                toml::parse(vm["config"].as<std::string>())
            );
            if (v.is_ok()) {
                config = v.unwrap();
            } else {
                printheader(stderr);
                fmt::println(stderr, "An error occurred processing the configuration file");
                fmt::println(stderr, "{}", v.unwrap_err());

                boost::mpi::broadcast(world, ok, 0);
                return -1;
            }
        }

        if (vm.count("mset")) {
            // msets on the command line take precdence over those the config file
            config.msets = vm["mset"].as<std::vector<std::string>>();
        }

        if (config.msets.size()) {
            std::vector<casacore::MeasurementSet> msets;
            for (auto& fname : config.msets) msets.push_back({fname});

            // Load just the table metadata and perform validation across all
            // measurement sets
            DataTable tbl(msets, {
                .chanlow=config.chanlow, .chanhigh=config.chanhigh,
                .datacolumn=config.datacolumn, .phasecenter=config.phasecenter,
                .skipdata=true
            });

            // Set values to replace any default values (e.g. 0 for chanhigh, empty phase)
            config.datacolumn = tbl.datacolumn();
            config.chanlow = tbl.chanlow();
            config.chanhigh = tbl.chanhigh();
            config.phasecenter = tbl.phasecenter();
        } else {
            printheader(stderr);
            fmt::println(stderr, "No msets provided; doing nothing");

            boost::mpi::broadcast(world, ok, 0);
            return -1;
        }

        // Validate the configuration file
        try {
            config.validate();
        } catch (const std::runtime_error& e) {
            printheader(stderr);
            fmt::println(stderr, "An error occurred validating the configuration");
            std::cerr << e.what() << std::endl;

            boost::mpi::broadcast(world, ok, 0);
            return -1;
        }

        if (world.size() != config.channelsOut + 1) {
            printheader(stderr);
            fmt::println(
                stderr,
                "Error: pigi must be started with (1 + channelsout={}) MPI processes; {} detected",
                config.channelsOut, world.size()
            );

            boost::mpi::broadcast(world, ok, 0);
            return -1;
        }

        // Print config to stdout and continue with Pigi workflow
        fmt::println("########## Configuration ##########");
        std::cout << std::setw(80) << toml::value(config) << std::endl;
        fmt::println("##################################");
        ok = true;
    }

    // Check to see if rank=0 made it past command line processing
    boost::mpi::broadcast(world, ok, 0);
    if (!ok) return 0; // Child processes return 0, irrespective of root process

    // Broadcast the configuration
    boost::mpi::broadcast(world, config, 0);

    // Set logging level
    Logger::setLevel(config.loglevel);

    auto local = world.split(world.rank() == 0);
    boost::mpi::intercommunicator intercom(local, 0, world, world.rank() == 0);

    if (world.rank() == 0) {
        if (config.precision == 32) {
            routines::cleanQueen<float>(config, intercom);
        } else {
            routines::cleanQueen<double>(config, intercom);
        }
    } else {
        if (config.precision == 32) {
            routines::cleanWorker<float>(config, intercom, local);
        } else {
            routines::cleanWorker<double>(config, intercom, local);
        }
    }
}