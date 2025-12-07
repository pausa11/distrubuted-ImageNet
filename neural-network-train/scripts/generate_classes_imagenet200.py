
import os

def main():
    wnids_path = 'data/tiny-imagenet-200/wnids.txt'
    output_path = 'src/tiny_imagenet_classes.py'
    
    if not os.path.exists(wnids_path):
        print(f"Error: {wnids_path} not found")
        return

    with open(wnids_path, 'r') as f:
        wnids = [line.strip() for line in f if line.strip()]
        
    wnids.sort()
    
    content = f"""# Generated from {wnids_path}
TINY_IMAGENET_SYNSETS = {repr(wnids)}
"""
    
    with open(output_path, 'w') as f:
        f.write(content)
        
    print(f"Successfully created {output_path} with {len(wnids)} classes")

if __name__ == '__main__':
    main()
