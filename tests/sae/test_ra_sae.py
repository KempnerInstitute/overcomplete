import torch
from overcomplete.sae import RATopKSAE, RAJumpSAE

def test_ra_implementations():
    # Mock data
    batch_size = 32
    input_dim = 128
    nb_concepts = 20
    
    #RelaxedArchetypalDictionary usually expects points to be [N, input_dim] ???
    points = torch.randn(100, input_dim) 
    input_data = torch.randn(batch_size, input_dim)

    print("Testing RATopKSAE...")
    ra_topk = RATopKSAE(
        input_shape=input_dim, 
        nb_concepts=nb_concepts, 
        points=points, 
        top_k=5
    )
    
    z_pre, z, x_hat = ra_topk(input_data)
    print(f"TopK Output shape: {x_hat.shape}")
    assert x_hat.shape == input_data.shape

    print("\nTesting RAJumpSAE...")
    ra_jump = RAJumpSAE(
        input_shape=input_dim, 
        nb_concepts=nb_concepts, 
        points=points,
        bandwith=0.001
    )
    
    z_pre, z, x_hat = ra_jump(input_data)
    print(f"Jump Output shape: {x_hat.shape}")
    assert x_hat.shape == input_data.shape
    
    print("\nSuccess! Both RA classes instantiated and forwarded.")

if __name__ == "__main__":
    test_ra_implementations()