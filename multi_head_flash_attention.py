from torch import nn
import torch

class MultiHeadFlashAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, block_size: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim必须能被num_heads整除"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.bc = block_size
        self.br = min(block_size, self.head_dim)
        self.scale = (self.head_dim ** -0.5)  # 添加缩放因子
        
    def forward(self, x: torch.Tensor, enable_flash: bool = False) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        if enable_flash:
            attn_output = self._flash_attn_multihead(q, k, v)
        else:
            attn_output = self._normal_attn_multihead(q, k, v)
            
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        return self.out_proj(attn_output)
    
    def _normal_attn_multihead(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q, k, v: [batch_size, num_heads, seq_len, head_dim]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # 使用统一的缩放因子
        attn_weights = torch.nn.functional.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)
    
    def _flash_attn_multihead(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        # q, k, v: [batch_size, num_heads, seq_len, head_dim]
        batch_size, num_heads, seq_len, head_dim = q.shape
        
        # 将批次和头维度合并以实现并行处理
        # 从 [batch_size, num_heads, seq_len, head_dim] 变为 [batch_size*num_heads, seq_len, head_dim]
        q_flat = q.reshape(batch_size * num_heads, seq_len, head_dim)
        k_flat = k.reshape(batch_size * num_heads, seq_len, head_dim)
        v_flat = v.reshape(batch_size * num_heads, seq_len, head_dim)
        
        # 创建所有头的输出、行和m变量
        output_flat = torch.zeros_like(q_flat)
        l_flat = torch.zeros(batch_size * num_heads, seq_len, device=q.device)
        m_flat = torch.full((batch_size * num_heads, seq_len), float('-inf'), device=q.device)
        
        # 并行处理所有头
        result_flat = self._flash_attn_parallel(q_flat, k_flat, v_flat, output_flat, l_flat, m_flat)
        
        # 恢复原始形状 [batch_size*num_heads, seq_len, head_dim] -> [batch_size, num_heads, seq_len, head_dim]
        result = result_flat.reshape(batch_size, num_heads, seq_len, head_dim)
        
        return result
    
    def _flash_attn_parallel(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           output: torch.Tensor, l: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
        """
        并行处理所有头的flash attention
        
        Args:
            q: [batch_size*num_heads, seq_len, head_dim]
            k: [batch_size*num_heads, seq_len, head_dim]
            v: [batch_size*num_heads, seq_len, head_dim]
            output: [batch_size*num_heads, seq_len, head_dim]
            l: [batch_size*num_heads, seq_len]
            m: [batch_size*num_heads, seq_len]
            
        Returns:
            result: [batch_size*num_heads, seq_len, head_dim]
        """
        batch_head_size, seq_len, head_dim = q.shape
        
        # 分块处理所有查询和键值对
        q_blocks = q.split(self.br, dim=1)  # 将所有批次的查询一起分块
        k_blocks = k.split(self.bc, dim=1)  # 将所有批次的键一起分块
        v_blocks = v.split(self.bc, dim=1)  # 将所有批次的值一起分块
        
        # 创建结果容器
        results = []
        
        # 为每个批次和头初始化累积变量
        o_accum = torch.zeros_like(q)
        l_accum = torch.zeros(batch_head_size, seq_len, device=q.device)
        m_accum = torch.full((batch_head_size, seq_len), float('-inf'), device=q.device)
        
        # 对每个块进行处理
        for j, (k_block, v_block) in enumerate(zip(k_blocks, v_blocks)):
            k_len = k_block.size(1)
            
            for i, q_block in enumerate(q_blocks):
                q_len = q_block.size(1)
                q_start = i * self.br
                q_end = min(q_start + q_len, seq_len)
                
                # 计算当前块的注意力分数，所有批次和头一起计算
                s = torch.bmm(q_block, k_block.transpose(1, 2)) * self.scale  # [batch*heads, q_len, k_len]
                
                # 获取当前分数的最大值
                m_ij, _ = torch.max(s, dim=2, keepdim=True)  # [batch*heads, q_len, 1]
                
                # 计算指数和归一化因子
                p_ij = torch.exp(s - m_ij)  # [batch*heads, q_len, k_len]
                l_ij = torch.sum(p_ij, dim=2)  # [batch*heads, q_len]
                
                # 提取当前查询块的累积变量
                m_i = m_accum[:, q_start:q_end]  # [batch*heads, q_len]
                l_i = l_accum[:, q_start:q_end]  # [batch*heads, q_len]
                o_i = o_accum[:, q_start:q_end]  # [batch*heads, q_len, head_dim]
                
                # 更新最大值和归一化因子
                m_new = torch.maximum(m_ij.squeeze(2), m_i)  # [batch*heads, q_len]
                l_new = torch.exp(m_i - m_new) * l_i + torch.exp(m_ij.squeeze(2) - m_new) * l_ij  # [batch*heads, q_len]
                
                # 重新加权累积输出并添加新的贡献
                # 处理分母为0的情况
                safe_l_new = torch.where(l_new > 0, l_new, torch.ones_like(l_new))
                
                # 计算重新加权的旧输出
                o_scale = (l_i / safe_l_new).unsqueeze(2)  # [batch*heads, q_len, 1]
                o_shift = torch.exp((m_i - m_new).unsqueeze(2))  # [batch*heads, q_len, 1]
                o_old = o_scale * o_shift * o_i  # [batch*heads, q_len, head_dim]
                
                # 计算新的贡献
                p_scale = torch.exp(m_ij - m_new.unsqueeze(2)) / safe_l_new.unsqueeze(2)  # [batch*heads, q_len, 1]
                o_new = torch.bmm(p_ij * p_scale, v_block)  # [batch*heads, q_len, head_dim]
                
                # 更新累积变量
                o_accum[:, q_start:q_end] = o_old + o_new
                m_accum[:, q_start:q_end] = m_new
                l_accum[:, q_start:q_end] = l_new
        
        return o_accum


if __name__ == "__main__":
    # 测试代码
    batch_size = 2
    seq_len = 128
    embed_dim = 512
    num_heads = 8
    
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    
    multi_attn = MultiHeadFlashAttention(embed_dim=embed_dim, num_heads=num_heads, block_size=32)
    for param in multi_attn.parameters():
        param.requires_grad = False
        
    x = torch.randn(batch_size, seq_len, embed_dim)
    
    # 先计算普通注意力
    normal_output = multi_attn(x, enable_flash=False)
    # 再计算flash attention
    flash_output = multi_attn(x, enable_flash=True)
    
    print("*" * 50 + "flash输出形状" + "*" * 50)
    print(flash_output.shape)
    print("*" * 50 + "普通输出形状" + "*" * 50)
    print(normal_output.shape)
    print("*" * 50 + "两种实现是否接近" + "*" * 50)
    print(torch.allclose(flash_output, normal_output, atol=1e-3))
    
    # 如果不相等，打印差异程度
    if not torch.allclose(flash_output, normal_output, atol=1e-3):
        print("最大差异:", torch.max(torch.abs(flash_output - normal_output)))
        print("平均差异:", torch.mean(torch.abs(flash_output - normal_output)))