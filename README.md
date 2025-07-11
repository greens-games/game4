### Data Storage thoughts
- #simd arrays in slices 
    - summing is difficult
    - Would have to find an easy way to transpose to another SIMD equiv ideally without converting
- Multiple n,m builtin matrices 
    - keeping track and cumulnation of results is difficult
    - I think this is last resort, could be very fast but the implementation is a nightmare
        - can't go past 4x4 on anything OR
        - A nightmare to do things past 4x4
        - to do this you would make your first X number of calcs, then your next X number of calcs and add them all up at the end
- #soa or aos
    - not sure yet haven't tried

#### NOTE: see playground for tests

### Idea
- Simple Neural network implementation
- Generational learning?
- Society simulator
- Raylib

### TODO:
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

