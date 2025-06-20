package main

import "core:fmt"
import "core:mem"
import "core:math"
import "core:math/linalg"
import "core:math/rand"

Classification :: enum {
	A,
	B,
	C,
	D,
	E,
}

main :: proc() {

	when ODIN_DEBUG {
		track: mem.Tracking_Allocator
		mem.tracking_allocator_init(&track, context.allocator)
		defer {
			if len(track.allocation_map) > 0 {
				fmt.eprintf("=== %v allocations not free: ===\n", len(track.allocation_map))
				for _, entry in track.allocation_map {
					fmt.eprintf("- %v bytes @ %v\n", entry.size, entry.location)
				}
			}
			mem.tracking_allocator_destroy(&track)
		}
	}
	rand.reset(1)
	//Process input
	input := [4]f64 {1.,0.,2., 6.}
	bias:f64 = rand.float64_range(0.,10.)

	//Init layers
	
	layer1 := create_matrix(4,4)
	for r in 0..<len(layer1) {
		for c in 0..<len(layer1[r]) {
			layer1[r][c] = rand.float64_range(0.,5.)
		}
	}


	/* layer1: matrix[4,4]f64 
	layer1 = {
		rand.float64_range(0.,5.), rand.float64_range(0.,5.), rand.float64_range(0.,5.), rand.float64_range(0.,5.),
		rand.float64_range(0.,5.), rand.float64_range(0.,5.), rand.float64_range(0.,5.), rand.float64_range(0.,5.),
		rand.float64_range(0.,5.), rand.float64_range(0.,5.), rand.float64_range(0.,5.), rand.float64_range(0.,5.),
		rand.float64_range(0.,5.), rand.float64_range(0.,5.), rand.float64_range(0.,5.), rand.float64_range(0.,5.),
	} 

	 output_layer_weights :matrix[4,len(Classification)]f64
	output_layer_weights = {
		rand.float64_range(0.,5.), rand.float64_range(0.,5.), rand.float64_range(0.,5.), rand.float64_range(0.,5.),
		rand.float64_range(0.,5.), rand.float64_range(0.,5.), rand.float64_range(0.,5.), rand.float64_range(0.,5.),
		rand.float64_range(0.,5.), rand.float64_range(0.,5.), rand.float64_range(0.,5.), rand.float64_range(0.,5.),
		rand.float64_range(0.,5.), rand.float64_range(0.,5.), rand.float64_range(0.,5.), rand.float64_range(0.,5.),
	} */

	output_layer_weights := create_matrix(len(Classification), 4)
	for r in 0..<len(output_layer_weights ) {
		for c in 0..<len(output_layer_weights[r]) {
			output_layer_weights[r][c] = rand.float64_range(0.,5.)
		}
	}
	fmt.println(layer1) 
	fmt.println()
	fmt.println(output_layer_weights)

	
	//Calculate layers
	layer1_output := dot(input[:], layer1)
	/* layer1_output := input * linalg.transpose(layer1) */
	for &n in layer1_output {
		n += bias
		n = relu(n)
	}
	fmt.println()
	fmt.println(layer1_output)
	
	output_layer := dot(layer1_output, output_layer_weights)
	/* output_layer := layer1_output * linalg.transpose(output_layer_weights) */
	for &n in output_layer {
		n += bias
	}

	fmt.println()
	fmt.println(output_layer)
	o := soft_max(output_layer[:])
	fmt.println()
	fmt.println(o)

	//Make decision
	sum:f64 = 0.
	chosen_output := -1
	curr_max := -1.
	for v, i in o {
		if v > curr_max {
			curr_max = v
			chosen_output = i
		}
		sum += v
	}
	fmt.println()
	fmt.println(sum)

	//Calc loss
	if chosen_output > -1 {
		fmt.println(cast(Classification) chosen_output)
		loss := cross_entropy_loss(o[chosen_output])
	}
	//Back Prop

	//Find loss at each neuron weight * loss at output * partial derivative of activation function for that neuron

	//Update weights:
	//new_weight := old_weight - alpha * (Z * delta) WHERE alpha is your learning rate; Z is the loss of the current neuron and delta is the loss of the previous linked neuron
}

weighted_sum :: proc(input, weights: []f64, bias: f64) -> f64 {
	assert(len(input) == len(weights))
	sum:f64 = 0.
	for i in 0..<len(input) {
		sum += input[i] * weights[i]
	}
	return sum + bias
}

relu :: proc(z: f64) -> f64 {
	return max(0., z)
}

d_relu :: proc(z: f64) -> f64 {
	return z >= 0 ? 1 : 0
}

soft_max :: proc(x: []f64) -> [len(Classification)]f64{
	assert(len(x) == len(Classification))
	//For a given vector calculate:
	//sum of math.exp of each value
	//Then for each value it is exp(i)/ sum of exp(i)
	ret:[len(Classification)]f64
	exp_sum:f64 = 0.
	for val in x {
		exp_sum += math.exp(val)
	}
	for val, index in x {
		ret[index] = math.exp(val)/exp_sum
	}
	return ret
}

calc_loss :: proc() {
	//
}

//Normally uses one-hot encoding to get the loss
//sum from 1 - number of classifications of y*log(y-hat) where y is the one-hot encoded entry and y-hat is the chosen class from prediction
cross_entropy_loss :: proc(predicted_val: f64) -> f64 {
	return -(math.log10(predicted_val))
}

dot :: proc(v: []f64, m:[][]f64) -> []f64 {
	assert(len(v) == len(m[0]))
	output := make_slice([]f64, len(m))
	
	for r in 0..<len(m) {
		curr_sum := 0.
		for c in 0..<len(m[r]) {
			curr_sum += v[c] * m[r][c]
		}
		output[r] = curr_sum
	}

	return output
}

create_matrix :: proc(rows, cols: int) -> [][]f64 {
	m := make_slice([][]f64 , rows)
	for &row in m {
		row = make_slice([]f64, cols)
	}
	return m 
}
