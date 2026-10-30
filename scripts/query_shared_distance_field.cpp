#include "care_collision_cdf/shared_distance_field.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <boost/property_tree/json_parser.hpp>
#include <boost/property_tree/ptree.hpp>

int main(int argc, char** argv) {
  try {
    std::string library, units, input_file, requested;
    if (argc == 5 && std::string(argv[1]) == "--catalog") {
      requested=argv[3];
      if (requested!="cdf" && requested!="sdf") throw std::invalid_argument("Backend must be cdf or sdf");
      boost::property_tree::ptree config;
      boost::property_tree::read_json(argv[2],config);
      library=config.get<std::string>("backends."+requested+".library");
      units=config.get<std::string>("backends."+requested+".output_units");
      if (library.empty() || library[0]!='/') throw std::invalid_argument("Catalog library paths must be absolute");
      input_file=argv[4];
    } else if (argc==4) {
      library=argv[1];units=argv[2];input_file=argv[3];
    } else throw std::invalid_argument("Usage: query_shared_distance_field LIBRARY EXPECTED_UNITS INPUT_ROWS.txt OR --catalog BACKENDS.json cdf|sdf INPUT_ROWS.txt");
    care_collision_cdf::SharedDistanceField field(library, units);
    if ((!requested.empty()) && field.backend() != (requested=="sdf" ? "rdf" : "cdf"))
      throw std::invalid_argument("Catalog backend kind mismatch");
    std::ifstream file(input_file);
    if (!file) throw std::invalid_argument("Cannot open input rows");
    std::vector<std::array<double, 10>> pairs;
    std::string line;
    while (std::getline(file, line)) {
      if (line.empty()) continue;
      std::istringstream input(line); std::array<double, 10> row;
      for (auto& x : row) if (!(input >> x)) throw std::invalid_argument("Expected exactly 10 values per row");
      std::string extra; if (input >> extra) throw std::invalid_argument("Extra input columns");
      pairs.push_back(row);
    }
    std::cerr << field.backend() << " units=" << field.units() << " pairs=" << pairs.size() << '\n';
    std::cout << std::setprecision(17);
    for (const auto& result : field.evaluate(pairs)) {
      std::cout << result.distance;
      for (double g : result.gradient) std::cout << ' ' << g;
      std::cout << '\n';
    }
    return 0;
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
