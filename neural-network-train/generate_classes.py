import os

def generate_classes_file():
    input_file = "imagenet_dirs.txt"
    output_file = "src/imagenet_classes.py"
    
    synsets = []
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            
            # Extract synset from path: gs://.../n01440764/ -> n01440764
            parts = line.split('/')
            # The path ends with /, so the synset is the second to last element
            # e.g. ['', ..., 'n01440764', '']
            if parts[-1] == '':
                synset = parts[-2]
            else:
                synset = parts[-1]
                
            if synset.startswith('n'):
                synsets.append(synset)
    
    synsets.sort()
    
    with open(output_file, 'w') as f:
        f.write("IMAGENET_SYNSETS = [\n")
        for s in synsets:
            f.write(f"    '{s}',\n")
        f.write("]\n")
        
    print(f"Generated {output_file} with {len(synsets)} classes.")

if __name__ == "__main__":
    generate_classes_file()
