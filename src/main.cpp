// Copyright(c) 2019 Federico Bolelli
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met :
// 
// *Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// 
// * Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and / or other materials provided with the distribution.
// 
// * Neither the name of "OpenCV_Project_Generator" nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED.IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


#include <experimental/filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

#define UPPER_BOUND_8_CONNECTIVITY ((size_t)((img_.rows + 1) / 2) * (size_t)((img_.cols + 1) / 2) + 1)

using namespace std::experimental::filesystem;
using namespace std;
using namespace cv;

class SAUF {
public:
	cv::Mat1b img_;
	cv::Mat1i img_labels_;
	unsigned n_labels_;
	string base_filename = "eq";
	ofstream os;
	bool os_good;

	SAUF() {}

	// Risale alla radice dell'albero a partire da un suo nodo n
	unsigned Find(const int *s_buf, unsigned n) {
		// Attenzione: non invocare la find su un pixel di background
		unsigned label = s_buf[n];

		assert(label > 0);

		while (label - 1 != n) {
			n = label - 1;
			label = s_buf[n];

			assert(label > 0);
		}
		return n;
	}


	// Unisce gli alberi contenenti i nodi a e b, collegandone le radici
	void Union(int *s_buf, unsigned a, unsigned b) {

		bool done;

		do {
			a = Find(s_buf, a);
			b = Find(s_buf, b);

			if (a < b) {
				int old = s_buf[b];
				s_buf[b] = min((unsigned)s_buf[b], a + 1);
				//int old = atomicMin(s_buf + b, a + 1)
				done = (old == b + 1);
				b = old - 1;
			}
			else if (b < a) {
				int old = s_buf[a];
				s_buf[a] = min((unsigned)s_buf[a], b + 1);
				done = (old == a + 1);
				a = old - 1;
			}
			else {
				done = true;
			}

		} while (!done);
	}

	void Flatten() {

		const int h = img_.rows;
		const int w = img_.cols;

		for (int r = 0; r < h; ++r) {
			for (int c = 0; c < w; ++c) {

				if (img_(r, c)) {
					img_labels_(r, c) = Find(reinterpret_cast<int *>(img_labels_.data), r * w + c) + 1;
				}
			}
		}
	}

	void DrawEquivalenceTrees(const string& filename_suffix) {

		path output_path = "../output";
		
		string filename_dot = base_filename + "_" + filename_suffix + ".dot";
		string filename_pdf = base_filename + "_" + filename_suffix + ".pdf";

		path output_path_dot = output_path / path(filename_dot);
		path output_path_pdf = output_path / path(filename_pdf);

		if (!exists(output_path)) {
			if (!create_directory(output_path)) {
				cout << "Unable to create output folder.\n";
			}
		}

		os.open(output_path_dot.string());
		if (!os) {
			cout << "Unable to open output file '" + filename_dot + "' to generate dot code.";
			return;
		}

		os << "digraph tree{\n";

		const int h = img_.rows;
		const int w = img_.cols;

		for (int r = 0; r < h; ++r) {
			for (int c = 0; c < w; ++c) {
				if (img_(r, c) > 0) {
					int id = r * w + c + 1;
					os << "\n" << "node" << id << " [label = " << id << "];";
					if (img_labels_(r, c) != id) {
						os << "\n" << "node" << img_labels_(r, c) << " -> " << "node" << id << ";";
					}
				}
			}
		}

		os << "}\n";
		os.close();


		string dot_command = "..\\tools\\dot\\dot -Tpdf " + output_path_dot.string() + " -o " + output_path_pdf.string();

		if (0 != system(dot_command.c_str())) {
			cout << "Unable to generate '" + filename_pdf + "'. \n";
		}

		string rm_command = "del " + output_path_dot.string();

		if (0 != system(rm_command.c_str())) {
			cout << "Unable to remove '" + filename_dot + "'. \n";
		}
	}

	void PerformLabeling()
	{
		const int h = img_.rows;
		const int w = img_.cols;

		img_labels_ = cv::Mat1i(img_.size(), 0); // Allocation + initialization of the output image
		for (int r = 0; r < h; ++r) {
			for (int c = 0; c < w; ++c) {
				if (img_(r, c) > 0) {
					img_labels_(r, c) = r * img_labels_.step[0] / sizeof(int) + c + 1;
				}
			}
		}
		// Rosenfeld Mask
		// +-+-+-+
		// |p|q|r|
		// +-+-+-+
		// |s|x|
		// +-+-+

			// First scan
		for (int r = 0; r < h; ++r) {
			// Get row pointers
			unsigned char const * const img_row = img_.ptr<unsigned char>(r);
			unsigned char const * const img_row_prev = (unsigned char *)(((char *)img_row) - img_.step.p[0]);
			unsigned * const  img_labels_row = img_labels_.ptr<unsigned>(r);
			unsigned * const  img_labels_row_prev = (unsigned *)(((char *)img_labels_row) - img_labels_.step.p[0]);

			for (int c = 0; c < w; ++c) {
#define CONDITION_P c > 0 && r > 0 && img_row_prev[c - 1] > 0
#define CONDITION_Q r > 0 && img_row_prev[c] > 0
#define CONDITION_R c < w - 1 && r > 0 && img_row_prev[c + 1] > 0
#define CONDITION_S c > 0 && img_row[c - 1] > 0
#define CONDITION_X img_row[c] > 0

				if (CONDITION_X) {
					// Supponendo che i dati della mat siano contigui
					if (CONDITION_P) {
						Union(reinterpret_cast<int*>(img_labels_.data), r * w + c, (r - 1) * w + c - 1);
					}

					if (CONDITION_Q) {
						Union(reinterpret_cast<int*>(img_labels_.data), r * w + c, (r - 1) * w + c);
					}

					if (CONDITION_R) {
						Union(reinterpret_cast<int*>(img_labels_.data), r * w + c, (r - 1) * w + c + 1);
					}

					if (CONDITION_S) {
						Union(reinterpret_cast<int*>(img_labels_.data), r * w + c, r * w + c - 1);
					}
				
					DrawEquivalenceTrees(to_string(r*w + c + 1));

				}
			}
		}
		
		Flatten();
	}

};

int main()
{
	// Read and display grayscale image
	Mat1b img_grayscale(8, 11);
	img_grayscale << 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0,
		0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0,
		0, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1,
		1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1,
		0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0;
	cv::threshold(img_grayscale, img_grayscale, 0, 1, cv::THRESH_BINARY_INV);

	SAUF s;
	s.img_ = img_grayscale;
	s.PerformLabeling();

	return EXIT_SUCCESS;
}
