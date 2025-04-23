#include <mpi.h>
#include <openssl/sha.h>
#include <filesystem>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <algorithm>

// #define LOG

namespace fs = std::filesystem;

bool compare_file_size(const fs::path& a, const fs::path& b) {
	return fs::file_size(a) < fs::file_size(b);
}

template<typename T>
std::vector<T> shuffle(const std::vector<T>& data, const int buckets) {
	int size = data.size();
	int fill = size / buckets + std::min(1, size % buckets);

	std::vector<T> shuffled;
	shuffled.reserve(data.size());

	for (int bucket = 0; bucket < buckets; bucket += 1) {
		for (int i = 0; i < fill; i += 1) {
			auto target = i * buckets + bucket;
			if (target < size) {
				shuffled.push_back(data[target]);
			}
		}
	}

	return shuffled;
}

void collect_files(const fs::path& path, std::vector<fs::path>& files) {
	if (!fs::exists(path)) {
		std::cerr << "Path does not exist: " << path << std::endl;
		return;
	}

	if (fs::is_regular_file(path)) {
		files.push_back(path);
	}
	else if (fs::is_directory(path)) {
		for (const auto& entry : fs::recursive_directory_iterator(path)) {
			if (fs::is_regular_file(entry)) {
				files.push_back(entry.path());
			}
		}
	}
}

std::string serialize(const std::vector<std::string> &strings) {
	std::ostringstream oss;
	for (const auto& s : strings) {
		oss << s << std::endl;
	}
	return oss.str();
}

std::vector<std::string> deserialize(const std::string &data) {
	std::vector<std::string> result;
	std::istringstream iss(data);
	std::string line;
	while (std::getline(iss, line)) {
		result.push_back(line);
	}
	return result;
}

std::string sha512(const std::vector<char> &data) {
	unsigned char hash[SHA512_DIGEST_LENGTH];
	SHA512(reinterpret_cast<const unsigned char*>(data.data()), data.size(), hash);

	std::ostringstream oss;
	for (int i = 0; i < SHA512_DIGEST_LENGTH; ++i) {
		oss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
	}
	return oss.str();
}

// According to this thread, this is one of the faster ways to read a file
// https://stackoverflow.com/a/525103
std::vector<char> file2bytes(const fs::path &path)
{
	std::ifstream ifs(path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
	if (ifs.fail()) {
		{
			std::stringstream ss;
			ss << "Failed to open file: " << path << std::endl;
			std::cerr << ss.str();
		}
		std::vector<char> empty;
		return empty;
	}

	std::ifstream::pos_type file_size = ifs.tellg();
	ifs.seekg(0, std::ios::beg);

	std::vector<char> bytes(file_size);
	ifs.read(bytes.data(), file_size);

	return bytes;
}

int main(int argc, char* argv[]) {
	MPI_Init(&argc, &argv);

	const int root_id = 0;
	int task_id, total_tasks;
	MPI_Comm_rank(MPI_COMM_WORLD, &task_id);
	MPI_Comm_size(MPI_COMM_WORLD, &total_tasks);

	std::vector<char> flat_root_files;
	std::vector<int> load(total_tasks);
	std::vector<int> displacement(total_tasks);

	// This is here and not in the `if`'s scope just because we need to use it later
	// We DO check that we are working with the root process, otherwise it will be empty
	std::vector<fs::path> root_files;

	if (task_id == root_id) {
		if (argc < 2) {
			std::cerr << "Usage: " << argv[0] << " <path1> [<path2> ...]" << std::endl;
			return 1;
		}

		for (int i = 1; i < argc; ++i) {
			fs::path input_path(argv[i]);
			collect_files(input_path, root_files);
		}
		int total_root_files = root_files.size();

#ifdef LOG
		std::stringstream ss;
		ss << "Root found " << total_root_files << " file(s):" << std::endl;
		for (const fs::path& path : root_files) {
			ss << "  " << path << std::endl;
		}
		ss << "---" << std::endl;
		std::cerr << ss.str() << std::endl;
#endif

		// This attempts to distribute the load as evenly as possible
		// by taking into account the size of files.
		// The next step in optimization would be to utilize the famous
		// backpack algorithm from dynamic programming.
		std::sort(root_files.begin(), root_files.end(), compare_file_size);
		root_files = shuffle(root_files, total_tasks);

		std::vector<std::vector<std::string>> partition(total_tasks);
		for (int i = 0; i < total_tasks; i += 1) {
			std::vector<std::string> p;
			p.reserve(total_root_files / total_tasks + 1);
			partition[i] = p;
		}
		for (int i = 0; i < total_root_files; i += 1) {
			partition[i % total_tasks].push_back(root_files[i].string());
		}

		int offset = 0;
		for (int i = 0; i < total_tasks; i += 1) {
			std::string s = serialize(partition[i]);
			load[i] = s.size();
			displacement[i] = offset;
			offset += s.size();
			flat_root_files.insert(flat_root_files.end(), s.begin(), s.end());
		}
	}

	int flat_files_length;
	MPI_Scatter(load.data(), 1, MPI_INT, &flat_files_length, 1, MPI_INT, 0, MPI_COMM_WORLD);

	std::vector<char> flat_files(flat_files_length);
	MPI_Scatterv(flat_root_files.data(), load.data(), displacement.data(), MPI_CHAR, flat_files.data(), flat_files_length, MPI_CHAR, 0, MPI_COMM_WORLD);

	std::string raw(flat_files.begin(), flat_files.end());
	std::vector<std::string> files = deserialize(raw);

#ifdef LOG
	{
		std::stringstream ss;
		ss << "Task #" << task_id << " received " << files.size() << " file(s):" << std::endl;
		for (const std::string& file : files) {
			ss << "  " << file << std::endl;
		}
		ss << "---" << std::endl;
		std::cerr << ss.str();
	}
#endif

	std::vector<std::string> hashed;
	hashed.reserve(files.size());
	for (const std::string& file : files) {
		fs::path path(file);
		auto text = file2bytes(path);
		if (text.size() == 0) {
			// failed to read the file
			hashed.push_back("");
		}
		else {
			auto hash = sha512(text);
#ifdef LOG
			std::stringstream ss;
			ss << "Task #" << task_id << " hashed file #" << hashed.size() << ": \'" << hash << "\'";
			std::cerr << ss.str() << std::endl;
#endif
			hashed.push_back(hash);
		}
	}

	std::string flat_hashed = serialize(hashed);

	int flat_hashed_length = flat_hashed.size();
	std::vector<int> flat_hashed_lengths, flat_hashed_displacements;

	if (task_id == root_id) {
		flat_hashed_lengths.resize(total_tasks);
	}

	MPI_Gather(&flat_hashed_length, 1, MPI_INT, flat_hashed_lengths.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

	std::vector<char> flat_root_hashed;
	if (task_id == root_id) {
		flat_hashed_displacements.resize(total_tasks);
		int offset = 0;
		for (int i = 0; i < total_tasks; ++i) {
			flat_hashed_displacements[i] = offset;
			offset += flat_hashed_lengths[i];
		}
		flat_root_hashed.resize(offset);
	}

	MPI_Gatherv(
		flat_hashed.data(), flat_hashed_length, MPI_CHAR,
		flat_root_hashed.data(), flat_hashed_lengths.data(), flat_hashed_displacements.data(), MPI_CHAR,
		0, MPI_COMM_WORLD
	);

	if (task_id == root_id) {
		std::string raw(flat_root_hashed.begin(), flat_root_hashed.end());
		std::vector<std::string> hashed = deserialize(raw);

		if (hashed.size() != root_files.size()) {
			std::cerr << "Mismatch between the number of hashes and the number of files (" << hashed.size() << " =!= " << root_files.size() << ")." << std::endl;
			std::cerr << "Displaying files and hashes separately." << std::endl;

			for (int i = 0; i < root_files.size(); i += 1) {
				std::cout << root_files[i].generic_string() << std::endl;
			}
			std::cout << std::endl;

			for (int i = 0; i < hashed.size(); i += 1) {
				std::cout << hashed[i] << std::endl;
			}
			std::cout << std::endl;
		}
		else {
			for (int i = 0; i < root_files.size(); i += 1) {
				std::cout << root_files[i].generic_string() << std::endl << hashed[i] << std::endl << std::endl;
			}
		}
	}

	MPI_Finalize();
	return 0;
}

