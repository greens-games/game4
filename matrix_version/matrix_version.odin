package matrix_version

import "core:terminal"
import "core:fmt" 
import "core:simd"
import "core:math/rand"
import "core:math/linalg"
import "core:math"
import "base:intrinsics"
import "core:time"
import "core:mem"
import "core:os"
import "core:strconv"

simd_16 :: #simd[16]f64
simd_8 :: #simd[8]f64
simd_4 :: #simd[4]f64

Classification :: enum {
	WORK,
	SLEEP,
	GATHER,
	TRAIN,
	/* FREE, */
}

Input_Layer :: struct {

}


Hidden_Layer :: struct {
	w1: matrix[4,4]f64,
	/* w2: matrix[4, 4]f64,
	w3: matrix[4, 4]f64,
	w4: matrix[4, 4]f64, */
	weighted_sums: simd_4,
	neurons: simd_4,
}

H_Layer :: struct {
	weights: matrix[4,4]f64,
	biases: matrix[1,4]f64,
	weighted_sums: matrix[1,4]f64,
	neurons: matrix[1,4]f64,
}

O_Layer_Classification :: struct {
	weights: matrix[4,4]f64,
	biases: matrix[1,4]f64,
	weighted_sums: matrix[1,4]f64,
	neurons: matrix[1,4]f64,
}

Output_Layer :: struct {
	w1: [4]simd_4,
	neurons: simd_4,
}

ITERATIONS :: 2
MAX_INPUT_VALUE :: 100.
MAX_WEIGHT_VALUE :: 2.
NUM_LAYERS :: 1
ALPHA :: 0.1

run :: proc() {
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
	input_matrix := make_slice([]simd_4, ITERATIONS)
	defer {
		delete(input_matrix)
	}

	expected_vector := make_slice([]Classification, ITERATIONS)
	defer delete(expected_vector)

	index := 0
	for i in 0..<ITERATIONS {
		input_matrix[i] = {
			rand.float64_range(0.,MAX_INPUT_VALUE),
			rand.float64_range(0.,MAX_INPUT_VALUE),
			rand.float64_range(0.,MAX_INPUT_VALUE),
			rand.float64_range(0.,MAX_INPUT_VALUE)
		}
		index, max := find_min(simd.to_array(input_matrix[i]))
		expected_vector[i] = Classification(index)
		index += 1
	}

	index = 0

	/* hidden_layer1, out_layer := test_layers() */

	hidden_layer1: Hidden_Layer
	hidden_layer1.w1 = random_matrix4x4()
	/* hidden_layer1.w2 = random_matrix4x4()
	hidden_layer1.w3 = random_matrix4x4()
	hidden_layer1.w4 = random_matrix4x4() */

	h_layer1: H_Layer
	h_layer1.weights = random_matrix4x4()
	h_layer1.biases = random_matrix1x4()
	o_layer: O_Layer_Classification
	o_layer.weights = random_matrix4x4()
	o_layer.biases = random_matrix1x4()

	hidden_layer2: Hidden_Layer
	hidden_layer2.w1 = random_matrix4x4()

	out_layer: Output_Layer
	for &weights in out_layer.w1 {
		weights = random_simd4()
	}

	hidden_layers:[NUM_LAYERS]^Hidden_Layer = {&hidden_layer1}

	timer: time.Stopwatch
	time.stopwatch_start(&timer)
	ret:[len(Classification)]f64
	for input, index in input_matrix {
		fmt.println("ITERATION: ", index)

		ret = forward_prop(input, &h_layer1, &o_layer, hidden_layers, &out_layer)
		fmt.println("RET: ", ret)
		fmt.println("EXPECTED_VECTOR: ", expected_vector[index])
		
		//TODO: Back prop
		
		back_prop(input, ret, &h_layer1, &o_layer, hidden_layers, &out_layer, expected_vector, index)
		//Cross entropy loss

		fmt.println()
	}
	time.stopwatch_stop(&timer)
	fmt.println(timer._accumulation)
	fmt.println("DONE TRAINING")

	/* running := true
	input_buff: [1024]byte
	data: [4]f64
	for running {
		
		fmt.print("WEALTH:")
		os.read(os.stdin, input_buff[:])
		data[0] = strconv.atof(string(input_buff[:3]))
		fmt.print("HEALTH:")
		os.read(os.stdin, input_buff[:])
		data[1] = strconv.atof(string(input_buff[:3]))

		fmt.print("SUPPLIES:")
		os.read(os.stdin, input_buff[:])
		data[2] = strconv.atof(string(input_buff[:3]))

		fmt.print("STENGTH:")
		os.read(os.stdin, input_buff[:])
		data[3] = strconv.atof(string(input_buff[:3]))

		simd_data := simd.from_array(data) 
		o := forward_prop(simd_data, &hidden_layer1, &out_layer)
		i, max := find_max(o)
		fmt.println(Classification(i))
		//TODO: Add some runtime training this means we need to figure out the expted value for a given state
	}  */
}

forward_prop :: proc(input: simd_4, h_layer1:^H_Layer, o_layer: ^O_Layer_Classification, hidden_layers: [NUM_LAYERS]^Hidden_Layer, out_layer: ^Output_Layer) -> [len(Classification)]f64 {
	_input := input
	_input = normalize_vector(_input)
	v := transmute(matrix[1,4]f64)_input

	h_layer1.weighted_sums = v * h_layer1.weights
	h_layer1.weighted_sums = h_layer1.weighted_sums + h_layer1.biases
	h_layer1.neurons = transmute(matrix[1,4]f64)relu(transmute(simd_4)h_layer1.weighted_sums)
	fmt.println("H_LAYER1: ", h_layer1)

	o_layer.weighted_sums = h_layer1.neurons * o_layer.weights
	o_layer.weighted_sums = h_layer1.weighted_sums + o_layer.biases
	o_layer.weighted_sums = transmute(matrix[1,4]f64)normalize_vector(transmute(simd_4)o_layer.weighted_sums)
	o_layer.neurons = transmute(matrix[1,4]f64)soft_max(transmute([len(Classification)]f64)o_layer.weighted_sums)

	//Soft max
	_ret := soft_max(simd.to_array(out_layer.neurons))
	ret := transmute([len(Classification)]f64)o_layer.neurons
	return ret
}

back_prop :: proc(input: simd_4, ret: [len(Classification)]f64, h_layer1: ^H_Layer, o_layer: ^O_Layer_Classification, hidden_layers: [NUM_LAYERS]^Hidden_Layer, out_layer: ^Output_Layer, expected_vector: []Classification, index: int) {
	/* temp_out_loss_v := simd.mul(out_loss_v, hidden_layers[0].neurons) */
	one_hot := [len(Classification)]f64{}
	//NOTE: Might still be able to use log(ret[int(expected_vector[index](]) instead of just 1 and everything else is 0
	one_hot[int(expected_vector[index])] = 1.
	dz2:matrix[1,4]f64 = transmute(matrix[1,4]f64)(ret - one_hot)

	//CHANGE IN OUTPUT LAYER
	temp_dz2 := transmute(f64)(dz2 * linalg.transpose(h_layer1.neurons))
	dw2 := 1/(ITERATIONS * temp_dz2)
	db2 := 1/(ITERATIONS * simd.reduce_add_bisect(transmute(simd_4)dz2))

	
	//CHANGE IN HIDDEN LAYER
	dz1 := linalg.matrix_comp_mul(linalg.transpose(o_layer.weights) * linalg.transpose(dz2), linalg.transpose(h_layer1.weighted_sums))
	_input := transmute(matrix[1,4]f64)input
	dw1 := 1/transmute(f64)(ITERATIONS * (linalg.transpose(dz1) * linalg.transpose(_input)))
	db1 := 1/(ITERATIONS * simd.reduce_add_bisect(transmute(simd_4)dz1))

	h_layer1.weights = h_layer1.weights - generate_matrix4x4(ALPHA * dw1)
	h_layer1.biases = h_layer1.biases - generate_matrix1x4(ALPHA * db1)
	
	o_layer.weights = o_layer.weights - generate_matrix4x4(ALPHA * dw2)
	o_layer.biases = o_layer.biases - generate_matrix1x4(ALPHA * db2)
}

soft_max :: proc(neurons: [len(Classification)]f64) -> [len(Classification)]f64 {
		ret: [len(Classification)]f64
		exp_sum:f64 = 0.
		for val in neurons {
			exp_sum += math.exp(val)
		}
		for val, index in neurons {
			ret[index] = math.exp(val)/exp_sum
		}
	return ret
}

dot_simd :: proc(v1, v2: $T) -> f64 {
	return simd.reduce_add_bisect(simd.mul(v1, v2))
}

relu :: proc(v: $T) -> T {
	zeros := T{}
	return simd.max(zeros, v)
}

d_relu_simd16 :: proc(v: simd_16) -> #simd [16]u64 {
	_v:#simd [16]u64 = auto_cast simd.floor(v)
	return simd.lanes_ge(_v , #simd [16]u64{})
}

d_relu_simd4 :: proc(v: simd_4) -> #simd [4]u64 {
	_v:#simd [4]u64 = auto_cast simd.floor(v)
	return simd.lanes_ge(_v , #simd [4]u64{})
}

d_relu :: proc(z: simd_4) -> simd_4 {
	arr := simd.to_array(z)
	for &val in arr {
		val = val >= 0. ? 1. : 0.
	}
	return simd.from_array(arr)
}

find_min :: proc(vector: [4]f64) -> (int, f64) {
	curr_i := -1
	curr_min := 99999999.
	for n, i in vector {
		if n < curr_min {
			curr_min = n
			curr_i = i
		}
	}
	return curr_i, curr_min
}

find_max :: proc(vector: [4]f64) -> (int, f64) {
	curr_i := -1
	curr_max := -99999999.
	for n, i in vector {
		if n > curr_max {
			curr_max = n
			curr_i = i
		}
	}
	return curr_i, curr_max
}

random_matrix4x4 :: proc() -> matrix[4,4]f64 {
	return {
	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),
	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),
	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),
	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),
	}
}

generate_matrix4x4 :: proc(val: f64) -> matrix[4,4]f64 {
	return {
	val,	val,	val,	val,
	val,	val,	val,	val,
	val,	val,	val,	val,
	val,	val,	val,	val,
	}
}

generate_matrix1x4 :: proc(val: f64) -> matrix[1,4]f64 {
	return {
	val,	val,	val,	val,
	}
}


random_matrix4x1 :: proc() -> matrix[4,1]f64 {
	return {
	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),
	}
}

random_matrix1x4 :: proc() -> matrix[1,4]f64 {
	return {
	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),
	}
}

random_simd16 :: proc() -> simd_16 {
	return {
	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),
	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),
	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),
	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),
	}
}

random_simd4 :: proc() -> simd_4 {
	return {
	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),	rand.float64_range(0.,MAX_WEIGHT_VALUE),
	}
}

generate_simd16 :: proc(val: f64) -> simd_16 {
	return {
	val, val, val, val,
	val, val, val, val,
	val, val, val, val,
	val, val, val, val,
	}
}

generate_simd4 :: proc(val: f64) -> simd_4 {
	return {
	val, val, val, val,
	}
}

normalize_simd16 :: proc(v: simd_16) -> simd_16 {
	_v := v
	_v = simd.sqrt(_v)
	_v = simd.div(_v, 16)
	return _v
}

normalize_simd4 :: proc(v: simd_4) -> simd_4 {
	_v := v
	_v = simd.sqrt(_v)
	_v = simd.div(_v, 4)
	return _v
}

//NOTE: This probably will need to change to match slow_version
normalize_simd :: proc(v: $T) -> T {
	_v := v
	_v = simd.sqrt(_v)
	_v = simd.div(_v, len(_v))
	return _v
}

normalize_vector :: proc(v: $T) -> T {
	vector := simd.to_array(v)
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
	for &val in vector {
		numerator := val - min
		range := 1. - -1.
		val = (numerator/denom) * range + 1
	}
	return simd.from_array(vector)
}

/* test_layers :: proc() -> (h_layer:Hidden_Layer, o_layer:Output_Layer) {


	h_layer.w1 = {-5.8565632650155068, -3.839782176429364, -1.9285955027797339, -3.5914623853645207, -1.1026883696258656, -3.498582399892622, -5.1930026589907108, -3.2482181030027655, -3.787649113347765, -3.9276615790684239, -5.498037256129967, -2.222753162113039, -3.6291536526131769, -2.5965786057963287, -5.8887698870855889, -1.1299246759544488}

	h_layer.w2 = {-2.4943036689915603, -3.667012675266214, -4.9799568461732537, -3.456556116593007, -2.5999483489585287, -4.0296499426013108, -3.1717312042922465, -4.4051875568073946, -2.973924219229199, -4.4332595327943976, -2.9968994867069139, -1.9581711713042713, -1.5716474475957254, -1.738294223340116, -2.8608819571265589, -3.4631042399271386}

	h_layer.w3 = {-5.956797996399081, -2.5291879259794916, -1.6056287067631603, -1.7442519296766754, -2.489154849421629, -2.350851005529792, -3.3519942207594475, -2.19616103178465, -5.3347565783825388, -1.5299008385484276, -3.915657725512842, -1.1779950344506922, -1.391551718844866, -2.940983227861703, -5.8916561485712506, -1.3977123282301696}

	h_layer.w4 = {-5.018456793359396, -3.4046069691836336, -1.6275292238306438, -5.402894362727134, -4.349700626244086, -2.0758871960101972, -5.118020098892444, -3.3349149754117753, -2.062355153566237, -1.117519623610308, -3.568757177583261, -4.60896882152438, -4.1305870876245239, -4.2005957528345395, -4.5480299086441285, -1.1897587327954202}

	o_layer = Output_Layer{
		w1 = {
			{4.507554829172683, 1.1854348611560506, 2.260804232892744, 4.5181871458235205, 1.4388439222010589, 0.37092651465489446, 2.8991956638983987, 4.4054929735526605, 4.313165532610239, 1.8801567734739588, 1.6376041642025667, 3.602636274215268, 4.35738374301488, 4.217137237823003, 0.07231215955381498, 0.35114279474454879}, {2.4341987959971796, 1.2060526891968664, 3.8236684238862808, 1.2619577676897868, 3.1041764884637595, 4.5424318230547618, 3.7365052146342714, 0.04742326446625522, 1.0817555840887303, 1.769427174910773, 4.7339440215406965, 0.07547911233856219, 2.8468694859274235, 3.7663867922630456, 3.85949744776881, 3.1858860412363392}, {3.7646925887651719, 0.5266002785552801, 1.7580830579211282, 3.3966488904863379, 0.037217000031373626, 3.5565620206445914, 0.5930114769610859, 3.275790619715443, 2.1906996061192276, 3.725761254027318, 3.3908823541226498, 3.4020550666501959, 0.96044132950758176, 4.3128668824536458, 4.5319769447391138, 4.674700036659721}, {3.147122641490687, 1.990979610981056, 0.69675322202380308, 2.9799229629045083, 2.2459856196870613, 3.7049264789024527, 4.240409333513722, 1.7369213164585033, 2.2690808056355305, 1.8149709760083386, 1.8991176031105959, -0.03556580053752878, 4.0184340087232489, 0.9552120744713732, 3.2105858888605003, 2.528274610093832}
		}
	}
	return
} */
