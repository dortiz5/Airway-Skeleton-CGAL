#include <CGAL/Simple_cartesian.h>
#include <CGAL/Surface_mesh.h>
#include <CGAL/Mean_curvature_flow_skeletonization.h>

#include <fstream>
#include <iostream>
#include <string>

typedef CGAL::Simple_cartesian<double>                         Kernel;
typedef Kernel::Point_3                                        Point;
typedef CGAL::Surface_mesh<Point>                              Triangle_mesh;
typedef boost::graph_traits<Triangle_mesh>::vertex_descriptor  vertex_descriptor;
typedef CGAL::Mean_curvature_flow_skeletonization<Triangle_mesh> Skeletonization;
typedef Skeletonization::Skeleton                              Skeleton;
typedef Skeleton::vertex_descriptor                            Skeleton_vertex;
typedef Skeleton::edge_descriptor                              Skeleton_edge;

int main(int argc, char* argv[])
{
    // Cambios aquí: input y output por argumentos
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input.off> <output_base>" << std::endl;
        return EXIT_FAILURE;
    }

    std::string input_path = argv[1];
    std::string output_base = argv[2];

    std::ifstream input(input_path.c_str());
    Triangle_mesh tmesh;
    input >> tmesh;
    if (!CGAL::is_triangle_mesh(tmesh))
    {
        std::cout << "Input geometry is not triangulated." << std::endl;
        return EXIT_FAILURE;
    }

    Skeleton skeleton;
    Skeletonization mcs(tmesh);

    // 1. Contract the mesh by mean curvature flow.
    mcs.contract_geometry();

    // 2. Collapse short edges and split bad triangles.
    mcs.collapse_edges();
    mcs.split_faces();

    // 3. Fix degenerate vertices.
    mcs.detect_degeneracies();

    // Perform the above three steps in one iteration.
    mcs.contract();

    // Iteratively apply step 1 to 3 until convergence.
    mcs.contract_until_convergence();

    // Convert the contracted mesh into a curve skeleton and
    // get the correspondent surface points
    mcs.convert_to_skeleton(skeleton);

    std::cout << "Number of vertices of the skeleton: " << boost::num_vertices(skeleton) << "\n";
    std::cout << "Number of edges of the skeleton: " << boost::num_edges(skeleton) << "\n";

    // Output all the edges of the skeleton.
    std::ofstream output((output_base + "_skel_poly.cgal"));
    std::set<Skeleton_edge> visited_edges;

    for (Skeleton_vertex v : CGAL::make_range(vertices(skeleton))) {
        int degree = out_degree(v, skeleton);
        if (degree != 2) { // extremo o bifurcación
            for (auto edge : CGAL::make_range(out_edges(v, skeleton))) {
                if (visited_edges.count(edge)) continue;
                std::vector<Point> polyline;
                Skeleton_vertex current = v;
                Skeleton_edge current_edge = edge;
                while (true) {
                    polyline.push_back(skeleton[current].point);
                    visited_edges.insert(current_edge);
                    Skeleton_vertex next = target(current_edge, skeleton);
                    if (out_degree(next, skeleton) != 2 || next == v) {
                        polyline.push_back(skeleton[next].point);
                        break;
                    }
                    // avanzar al siguiente edge
                    for (auto next_edge : CGAL::make_range(out_edges(next, skeleton))) {
                        if (!visited_edges.count(next_edge) && next_edge != current_edge) {
                            current_edge = next_edge;
                            current = next;
                            break;
                        }
                    }
                }
                // guardar polyline en archivo
                output << polyline.size();
                for (auto& p : polyline) output << " " << p;
                output << "\n";
            }
        }
    }

    // Output skeleton points and the corresponding surface points
    std::ofstream output_corr((output_base + "_correspondance_sm_polylines.txt").c_str());
    for(Skeleton_vertex v : CGAL::make_range(vertices(skeleton)))
        for(vertex_descriptor vd : skeleton[v].vertices)
            output_corr << "2 " << skeleton[v].point << " " << get(CGAL::vertex_point, tmesh, vd) << "\n";
    output_corr.close();

    return EXIT_SUCCESS;
}
