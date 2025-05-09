from torch import nn
import torch

class FlashAttention(nn.Module):
    def __init__(self, embed_dim: int, block_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.bc=block_size
        self.br=min(block_size,embed_dim)
        
    def forward(self, x:torch.Tensor, enable_flash: bool=False) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        output=torch.zeros_like(x)
        l=torch.zeros(x.shape[0])
        m=torch.Tensor([float('-inf') for i in range(x.shape[0])])
        if enable_flash:
            return self.flash_attn(q, k, v,output,l,m)
        else:
            return self.normal_attn(q, k, v)
        
        
    def normal_attn(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor) -> torch.Tensor:
        s=torch.matmul(q,k.transpose(0,1))
        softmaxed=torch.nn.functional.softmax(s,dim=1)
        return torch.matmul(softmaxed,v)
    
    def flash_attn(self, q:torch.Tensor, k:torch.Tensor, v:torch.Tensor,output:torch.Tensor,l:torch.Tensor,m:torch.Tensor) -> torch.Tensor:
        q_blocks=list(q.split(self.br, dim=0))
        k_blocks=list(k.split(self.bc, dim=0))
        v_blocks=list(v.split(self.bc, dim=0))
        o_blocks=list(output.split(self.br, dim=0))
        l_blocks=list(l.split(self.br, dim=0))
        m_blocks=list(m.split(self.br, dim=0))
        for j in range(len(k_blocks)):
            k_block=k_blocks[j]
            v_block=v_blocks[j]
            for i in range(len(q_blocks)):
                q_block=q_blocks[i]
                o_block=o_blocks[i]
                l_i=l_blocks[i]
                m_i=m_blocks[i]
                s=torch.matmul(q_block,k_block.transpose(0,1))
                m_ij,_=torch.max(s,dim=1,keepdim=True)
                p_ij=torch.exp(s-m_ij)
                l_ij=torch.sum(p_ij,dim=1)
                mi_new=torch.max(m_ij.squeeze(1),m_i)
                li_new=torch.exp(m_i-mi_new)*l_i+torch.exp(m_ij.squeeze(1)-mi_new)*l_ij
                o_block= (l_i/li_new).unsqueeze(1) * torch.exp((m_i - mi_new).unsqueeze(1)) * o_block + torch.matmul(p_ij * (torch.exp(m_ij - mi_new.unsqueeze(1)) / li_new.unsqueeze(1)), v_block)
                l_blocks[i]=li_new
                m_blocks[i]=mi_new
                o_blocks[i]=o_block
        return torch.cat(o_blocks,dim=0)
                

if __name__ == "__main__":
    attn=FlashAttention(embed_dim=1024,block_size=4)
    x=torch.randn(1024,1024)
    flash_atten=attn(x,enable_flash=True)   
    normal_atten=attn(x,enable_flash=False)
    print("*"*50+"flash_atten"+"*"*50)
    print(flash_atten)
    print(flash_atten.shape)
    print("*"*50+"normal_atten"+"*"*50)
    print(normal_atten)
    print(normal_atten.shape)
    print("*"*50+"is_close"+"*"*50)
    print(torch.allclose(flash_atten,normal_atten,atol=1e-3))
    