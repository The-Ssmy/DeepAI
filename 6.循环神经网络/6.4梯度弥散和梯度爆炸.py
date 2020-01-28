import torch

"""
梯度爆炸：
    可以对w的梯度进行缩放操作，将w的梯度的模进行缩放，方向保持不变
    注意是对w梯度的模进行缩放不是对w本身进行缩放
    
梯度弥散：
    


"""
loss = criterion(output, y)
mdoel.zero_grad()
loss.backward()
for p in model.parameters():
    print(p.grad.norm())
    torch.nn.utils.clip_grad_norm_(p, 10)  # 第一个参数是要传送的tensor，第二个参数是指要把grad的模缩放到10以内的范围
optimizer.step()









