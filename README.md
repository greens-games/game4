### Idea
- Simple Neural network implementation
- Generational learning?
- Society simulator
- Raylib

### TODO:
- Calc loss
- Back prop
- Multiple iterations
- Show better output
- Make non-classification decisions (like how predicting house prices would be)

### Most likely issues
- How to calculate loss without base training data/ expected results

### Some Notes

- Matrices
    - Rows is each vector weight going to the neuron
    - Col is the specific neuron
    - rows must = length of input vector 
    - cols = number of neurons can be anything 
    - Odin is limited ot 4x4 builtin 


### Testing
- data sets found in data_set
    - chars_data is data about fictional characters
    - pokemon_data is data about pokemon 
- The intention for these data sets is to test out the network and make sure it somewhat works on existing data sets
- These data_sets will need to be cleaned up to accomplish some goal
- Pokemon is already in csv format so that should be easy enough to start
- Chars would require atleast exporting to csv
