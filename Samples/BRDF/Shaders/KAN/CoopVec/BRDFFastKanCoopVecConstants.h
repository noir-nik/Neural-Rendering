#pragma once

static const int MAX_KAN_LAYERS =  4;

struct FastKanLayerBufferOffsets {
	uint32_t rbf_grid;
	uint32_t spline_weight;
	uint32_t base_weight;
	uint32_t base_bias;
};

struct FastKanConstants {
	int                       num_layers;
	int                       pad[3];
	FastKanLayerBufferOffsets offsets[MAX_KAN_LAYERS];
};
