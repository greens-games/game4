package graph

import "../constants"

import "core:fmt"
import "core:strconv"
import "core:strings"
import rl "vendor:raylib"

RADIUS :: 36.

Activation_Function :: enum {
	NONE,
	RELU,
	SOFT_MAX,
}

Edge :: struct {
	start_point: [2]f32,
	end_point: [2]f32,
}

Node :: struct {
	origin: [2]f32,
	activated: bool,
	value: f64,
}

Network :: struct {
	input_layer: []f64,
	hidden_layers: [][]f64,
	output_layer: []f64,
}

draw_hidden_layers :: proc(nodes: []Node) {
	denom: i32 = auto_cast len(nodes) + 1
	width_denom: i32 = 4
	height := rl.GetScreenHeight()
	width := rl.GetScreenWidth()
	for i in constants.NUM_LAYERS..< constants.NUM_LAYERS + 1 {
		for node, idx in nodes {
			if node.activated {
				rl.DrawCircle(width * i32(i + 1)/width_denom,  height * i32(idx + 1)/denom, RADIUS, rl.GREEN)
			} else {
				rl.DrawCircleLines(width * i32(i+ 1)/width_denom,  height * i32(idx + 1)/denom, RADIUS, rl.WHITE)
			}
		}
	}
}

draw_input_layer :: proc(nodes: []Node) {
	denom: i32 = auto_cast len(nodes) + 1
	width_denom: i32 = constants.NUM_LAYERS + 3
	height := rl.GetScreenHeight()
	width := rl.GetScreenWidth()
	for node, idx in nodes {
		text := fmt.ctprintf("%f", node.value)
		rl.DrawText(text, (width/width_denom) - i32(len(text) * 3),  height * i32(idx + 1)/denom, 12, rl.WHITE)
		rl.DrawCircleLines(width/width_denom,  height * i32(idx + 1)/denom, RADIUS, rl.WHITE)
	}
}

draw_output_layer :: proc(nodes: []Node) {
	denom: i32 = auto_cast len(nodes) + 1
	width_denom: i32 = constants.NUM_LAYERS + 3
	height := rl.GetScreenHeight()
	width := rl.GetScreenWidth()
	for node, idx in nodes {
		if node.activated {
			rl.DrawCircle(width * 3/width_denom,  height * i32(idx + 1)/denom, RADIUS, rl.GREEN)
		} else {
			rl.DrawCircleLines(width * 3/width_denom,  height * i32(idx + 1)/denom, RADIUS, rl.WHITE)
		}
	}
}

convert_to_nodes :: proc(neurons: []f64, a_function: Activation_Function) -> (nodes: [constants.H_NUM_NEURONS]Node) {
	max := -99999.
	max_idx := -1
	for neuron, idx in neurons {
		node: Node
		node.value = neuron
		switch a_function {
		case .NONE:
		case .RELU:
			if neuron > 0 {
				node.activated = true
			} else {
				node.activated = false
			}
		case .SOFT_MAX:
			if node.value > max {
				max = node.value
				max_idx = idx
			}
		}
		nodes[idx] = node
	}
	if a_function == .SOFT_MAX {
		nodes[max_idx].activated = true
	}
	return
}
