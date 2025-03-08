""" Some clear examples of pytorch funcitons.
"""
import torch
import torch.nn as nn


def gather_eg(in_tensor):
    """ Demonstrate the torch.gather function, as used in calculate_objectness_loss.
    Args:
        in_tensor (any): Dummy tensor.
    """
    in_size = tuple(in_tensor.shape)
    print("Input tensor shape: %s"%(in_size, ))
    print(f"Input tensor: {in_tensor.squeeze()}")
    
    label_tensor = torch.rand(size=in_size, dtype=in_tensor.dtype)
    print(f"Label tensor: {label_tensor.squeeze()}")
    
    random_indices = torch.randint(low=0, high=in_size[1]-1, size=in_size)  # These would be unique indices in real world
    print(f"Indices tensor: {random_indices.squeeze()}")
    
    criterion = nn.CrossEntropyLoss(reduction='mean')
    objectness_score = in_tensor
    objectness_label = label_tensor
    fp2_inds = random_indices
    objectness_label = torch.gather(objectness_label, 1, fp2_inds)  # Gathers values along axis specified by dim.
    print(f"Objectness label after gather: {objectness_label}")
    
    loss = criterion(objectness_score, objectness_label)
    print("Loss: %f"%loss)
    return
    
def argmax_eg(in_tensor):
    """ Demonstrate argmax function.
    Args:
        in_tensor (any): Dummy tensor.
    """
    in_size = tuple(in_tensor.shape)
    print("Input tensor shape: %s"%(in_size, ))
    print(f"Input tensor: {in_tensor.squeeze()}")
    
    max_indices = torch.argmax(in_tensor, 1)  # Index 1 means along the rows, i.e. for each entry in the row return the index that has the highest value
    print(f"Result from argmax: {max_indices}")
    
    return
    
if __name__ == '__main__':
    dummy_input_tensor = torch.rand((1, 2, 10))  # (B, object/not_object, N)
    while True:
        selection = input("'gather', this or that? ").strip()
        if selection == 'gather':
            gather_eg(dummy_input_tensor)
        elif selection == 'argmax':
            argmax_eg(dummy_input_tensor)
        else:
            break
        