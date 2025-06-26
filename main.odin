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
		context.allocator = mem.tracking_allocator(&track)
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
	input := []f64{1.,0.,2., 6.}
	input = normalize_vector(input[:])
	//find max and min
	bias:f64 = rand.float64_range(0.,10.)

	//Init layers
	
	layer1_weights := create_matrix(4,4)
	defer {
		for &row in layer1_weights {
			delete(row)
		}
		delete(layer1_weights)
	}
	for r in 0..<len(layer1_weights) {
		for c in 0..<len(layer1_weights[r]) {
			layer1_weights[r][c] = rand.float64_range(0.,5.)
		}
	}

	output_layer_weights := create_matrix(len(Classification), 4)
	defer {
		for &row in output_layer_weights {
			delete(row)
		}
		delete(output_layer_weights)
	}
	for r in 0..<len(output_layer_weights ) {
		for c in 0..<len(output_layer_weights[r]) {
			output_layer_weights[r][c] = rand.float64_range(0.,5.)
		}
	}
	
	//Calculate layers
	layer1_neurons := dot(input[:], layer1_weights)
	defer delete(layer1_neurons)
	for &n in layer1_neurons {
		n += bias
		n = relu(n)
	}
	layer1_neurons = normalize_vector(layer1_neurons[:])
	
	output_layer := dot(layer1_neurons, output_layer_weights)
	defer delete(output_layer )
	for &n in output_layer {
		n += bias
	}
	output_layer = normalize_vector(output_layer[:])
	o := soft_max(output_layer[:])

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

	//Calc loss
	fmt.println("OUTPUT VALUES:")
	fmt.println(o)
	if chosen_output > -1 {
		fmt.println(cast(Classification) chosen_output)
		fmt.println(chosen_output)
		fmt.println(curr_max)
		loss := cross_entropy_loss(o[chosen_output])

		//Back Prop

		//Find loss at each neuron weight * loss at output * partial derivative of activation function for that neuron
		fmt.println("\nOUTPUT LAYER WEIGHTS: ")
		for row in output_layer_weights {
			fmt.printfln("%v", row)
		}

		layer1_neuron_loss := make_slice([]f64, len(layer1_neurons))
		defer delete(layer1_neuron_loss)
		for neuron, n in layer1_neurons {
			weight := output_layer_weights[chosen_output][n]
			d_a := d_relu(neuron)
			neuron_loss := weight * loss * d_a
			layer1_neuron_loss[n] = neuron_loss
			fmt.printfln("PARAMS FOR NEURON %v = %v, %v, %v", n, loss, weight, d_a)
			fmt.printfln("LOSS OF NEURON %v = %v", n, neuron_loss)
		}

		fmt.println("\n LAYER 1 NEURONS\n",layer1_neurons)
		fmt.println("\n LAYER 1 NEURON LOSS \n",layer1_neuron_loss)

		alpha := 0.1
		output_layer_weights_t := transpose(output_layer_weights) 
		fmt.println()
		fmt.println(output_layer_weights)
		for row, r in output_layer_weights_t {
			for &col, c in output_layer_weights_t[r] {
				//Update weights:
				//new_weight := old_weight - alpha * (Z * delta) WHERE alpha is your learning rate; Z is the loss of the current neuron and delta is the loss of the previous linked neuron
				new_weight := col - alpha * (layer1_neuron_loss[r] * loss)
				col = new_weight
			}
		}
		output_layer_weights = transpose(output_layer_weights_t)
		fmt.println(output_layer_weights)

}

back_prop :: proc(weights_to_update: [][]f64, neurons: []f64, chosen_output: int, loss: f64) {
		//Back Prop

		//Find loss at each neuron weight * loss at output * partial derivative of activation function for that neuron
		fmt.println("\nOUTPUT LAYER WEIGHTS: ")
		for row in weights_to_update {
			fmt.printfln("%v", row)
		}

		layer1_neuron_loss := make_slice([]f64, len(neurons))
		defer delete(layer1_neuron_loss)
		for neuron, n in neurons {
			weight := weights_to_update[chosen_output][n]
			d_a := d_relu(neuron)
			neuron_loss := weight * loss * d_a
			layer1_neuron_loss[n] = neuron_loss
			fmt.printfln("PARAMS FOR NEURON %v = %v, %v, %v", n, loss, weight, d_a)
			fmt.printfln("LOSS OF NEURON %v = %v", n, neuron_loss)
		}

		fmt.println("\n LAYER 1 NEURONS\n",neurons)
		fmt.println("\n LAYER 1 NEURON LOSS \n",layer1_neuron_loss)

		alpha := 0.1
		weights_to_update_t := transpose(weights_to_update) 
		fmt.println()
		fmt.println(weights_to_update)
		for row, r in weights_to_update_t {
			for &col, c in weights_to_update_t[r] {
				//Update weights:
				//new_weight := old_weight - alpha * (Z * delta) WHERE alpha is your learning rate; Z is the loss of the current neuron and delta is the loss of the previous linked neuron
				new_weight := col - alpha * (layer1_neuron_loss[r] * loss)
				col = new_weight
			}
		}
		new_weights_to_update := transpose(weights_to_update_t)
	}
	
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
	return z >= 0. ? 1. : 0.
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

normalize_vector :: proc(vector: []f64) -> []f64 {
	vector := vector
	min := 99999.
	max := -99999.
	for val in vector {
		if val < min {
			min = val
		}

		if val > max {
			max = val
		}
	}

	denom := max - min
	fmt.println("====== NORMALIZATION ======")
	fmt.println("INPUT: ", vector)
	fmt.println("MIN: ", min)
	fmt.println("MAX: ", max)
	fmt.println("DENOM: ", vector)
	for &val in vector {
		numerator := val - min
		range := 1. - -1.
		val = (numerator/denom) * range + 1
	}
	fmt.println("OUTPUT: ", vector)
	fmt.println("==========================\n")
	return vector
}

create_matrix :: proc(rows, cols: int) -> [][]f64 {
	m := make_slice([][]f64 , rows)
	for &row in m {
		row = make_slice([]f64, cols)
	}
	return m 
}

transpose :: proc(input_matrix: [][]f64) -> [][]f64 {
	input_matrix_t := make_slice([][]f64, len(input_matrix[0]))
	for &row in input_matrix_t {
		row = make_slice([]f64, len(input_matrix))
	}

	for c in 0..<len(input_matrix[0]) {
		for r in 0..<len(input_matrix) {
			input_matrix_t[c][r] = input_matrix[r][c]
		}
	}
	return input_matrix_t
}

temp :: proc() {
	x := 500.
	z := math.exp_f64(x)
	fmt.println(z)
}
