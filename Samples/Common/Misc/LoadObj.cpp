module;

#include "tiny_obj_loader.h"
#include <iostream>

module SamplesCommon;
import :LoadObj;
import :MeshPrimitives;

using namespace mesh;

auto LoadObj(std::string_view obj_path) -> LoadObjResult {
	std::vector<Vertex>              vertices;
	std::vector<UVSphere::IndexType> indices;

	// OBJ
	std::string inputfile = obj_path.data();

	tinyobj::ObjReaderConfig reader_config;
	// reader_config.mtl_search_path = "./"; // Path to material files

	tinyobj::ObjReader reader;

	if (!reader.ParseFromFile(inputfile, reader_config)) {
		if (!reader.Error().empty()) {
			std::cerr << "TinyObjReader: " << reader.Error();
		}
		exit(1);
	}

	if (!reader.Warning().empty()) {
		std::cout << "TinyObjReader: " << reader.Warning();
	}

	auto& attrib    = reader.GetAttrib();
	auto& shapes    = reader.GetShapes();
	auto& materials = reader.GetMaterials();

	// sizes
	std::printf("vertices: %zu\n", attrib.vertices.size() / 3);
	std::printf("normals: %zu\n", attrib.normals.size() / 3);
	std::printf("texcoords: %zu\n", attrib.texcoords.size() / 2);

	for (u32 i = 0; i < attrib.vertices.size() / 3; i++) {
		Vertex vv = {
			.pos = {
				attrib.vertices[3 * i + 0] / decltype(attrib.vertices[0])(30.),
				attrib.vertices[3 * i + 1] / decltype(attrib.vertices[0])(30.) - 1.3f,
				attrib.vertices[3 * i + 2] / decltype(attrib.vertices[0])(30.),
			},
		};
		// if (attrib.normals.size() > 0) {
		// 	if
		// 		// vv.normal = {attrib.normals[3 * i + 0], attrib.normals[3 * i + 1], attrib.normals[3 * i + 2]};
		// 		vv
		// 		.normal[0] = attrib.normals[3 * i + 0];
		// 	vv.normal[1] = attrib.normals[3 * i + 1];
		// 	vv.normal[2] = attrib.normals[3 * i + 2];
		// }
		if (attrib.texcoords.size() > 0) {
			vv.u = attrib.texcoords[2 * i + 0];
			vv.v = attrib.texcoords[2 * i + 1];
		}
		vertices.push_back(vv);
	}

	// Loop over shapes
	for (size_t s = 0; s < shapes.size(); s++) {
		// Loop over faces(polygon)
		size_t index_offset = 0;
		for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
			size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);

			// push vertices

			// Loop over vertices in the face.
			for (size_t v = 0; v < fv; v++) {
				// access to vertex
				tinyobj::index_t idx = shapes[s].mesh.indices[index_offset + v];
				tinyobj::real_t  vx  = attrib.vertices[3 * size_t(idx.vertex_index) + 0];
				tinyobj::real_t  vy  = attrib.vertices[3 * size_t(idx.vertex_index) + 1];
				tinyobj::real_t  vz  = attrib.vertices[3 * size_t(idx.vertex_index) + 2];

				// Vertex vv = {.pos = {vx, vy, vz}};
				// Check if `normal_index` is zero or positive. negative = no normal data
				if (idx.normal_index >= 0) {
					tinyobj::real_t nx = attrib.normals[3 * size_t(idx.normal_index) + 0];
					tinyobj::real_t ny = attrib.normals[3 * size_t(idx.normal_index) + 1];
					tinyobj::real_t nz = attrib.normals[3 * size_t(idx.normal_index) + 2];

					// vv.normal = {nx, ny, nz};
					auto& vv     = vertices[shapes[s].mesh.indices[index_offset + v].vertex_index];
					vv.normal[0] = nx;
					vv.normal[1] = ny;
					vv.normal[2] = nz;
				}

				// Check if `texcoord_index` is zero or positive. negative = no texcoord data
				if (idx.texcoord_index >= 0) {
					tinyobj::real_t tx = attrib.texcoords[2 * size_t(idx.texcoord_index) + 0];
					tinyobj::real_t ty = attrib.texcoords[2 * size_t(idx.texcoord_index) + 1];

					// vv.u = tx;
					// vv.v = ty;
				}

				// Optional: vertex colors
				// tinyobj::real_t red   = attrib.colors[3*size_t(idx.vertex_index)+0];
				// tinyobj::real_t green = attrib.colors[3*size_t(idx.vertex_index)+1];
				// tinyobj::real_t blue  = attrib.colors[3*size_t(idx.vertex_index)+2];

				indices.push_back(shapes[s].mesh.indices[index_offset + v].vertex_index);
			}
			index_offset += fv;

			// per-face material
			shapes[s].mesh.material_ids[f];
		}
	}
	return {std::move(vertices), std::move(indices)};
}
