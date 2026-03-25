import torch
import torch.nn as nn
from torch.autograd import Function
from hylearn.lib.modules.batch import LBNforward
from hylearn.lib.modules.batch import LBNbackward
from hylearn.lib.modules.batch import dict_set
from hylearn.lib.modules.batch import *
import torch.nn.functional as F
class STE(Function):
    @staticmethod
    def forward(ctx, input, nbit):
        scale = 2**nbit - 1
        return torch.round(input*scale)/scale

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None # nbit with no grad
def STE_DEF(input,nbit):
    return STE.apply(input,nbit)

def Ste_quanti(input, nbit):
    input1 = torch.tanh(input)
    input2 = input1/ (2*torch.max(torch.abs(input1))) + 0.5
    input3 = 2*STE_DEF(input2,nbit)-1
    return input3

def cpt_size(input):
    flat_input = input.flatten(start_dim=1)
    input_size = flat_input.shape[1]
    # print(input_size)
    input_channel = input.shape[0]

    # print(input_channel)
    input_kernel = input.shape[1]
    # print(input_kernel)
    input_x_size = input.shape[2]
    # print(input_x_size)
    input_y_size = input.shape[3]
    return input_size,input_channel,input_kernel, input_x_size, input_y_size


class quan_a(Function):
    @staticmethod
    def forward(ctx,input,cita,nbit_a):
        out1 = 0.5*(torch.abs(input)-torch.abs(input-cita)+cita)
        scale = 2**nbit_a - 1
        scale_cita = scale/cita
        out2 =torch.round(out1 * scale_cita) / scale_cita
        ctx.save_for_backward(input,out1,cita)
        return out2

    @staticmethod
    def backward(ctx,grad_output):
        grad_in,grad_cita,grad_x = None,None,None
        input,out1,cita= ctx.saved_tensors#  need , to keep tensor
        input_iter = input.clone().detach()
        input_boolen_1 = torch.ge(input_iter,0)
        input_boolen_2 = torch.lt(input_iter,cita)
        mask1 = input_boolen_1 * input_boolen_2
        mask2 = torch.ge(input_iter,cita)

        grad_in = grad_output*mask1
        grad_cita  = torch.Tensor([torch.sum(grad_output*mask2)]).to(device)
        return grad_in,grad_cita,None


def quan_a_fn(input,cita,nbita):
    return quan_a.apply(input,cita,nbita)


class L_STE(Function):
    @staticmethod
    def forward(ctx,x,weight2,linear0,gamma1,beta1,bn1,nbit,size,channel,kernel,x_size, y_size,orign_list,inter_list,MODE):
        scale = 2**nbit - 1
        out1 = x
        out1_tanh = torch.tanh(x)
        out2 = out1.flatten(start_dim =1).mm(linear0)
        out3,cache1 = LBNforward(out2,gamma1,beta1,bn1,MODE)
        out9 = out3
        out9_a = out9/ (2*torch.max(torch.abs(out9))) + 0.5
        out10 = out9_a
        output = 2* torch.round(out10*scale)/scale -1
        ctx.other = channel,kernel,x_size,y_size,cache1
        orign_list[weight2] = out1.detach()
        inter_list[weight2] = out9.detach()

        ctx.save_for_backward(x, linear0,out10,out9_a,out9,out3,out2,out1,out1_tanh)
        if kernel>0 and y_size>0 and x_size>0:
            output = output.reshape(channel,kernel,x_size,y_size)
        else:
            output = output
        return output
    @staticmethod
    def backward(ctx,grad_output):
        grad_out7,grad_out6,grad_out5,grad_out4,grad_out3,grad_linear2,grad_out2,grad_linear1,grad_out1,grad_linear0 = None,None,None,None,None,None,None,None,None,None
        grad_gamma4,grad_gamma3,grad_gamma2,grad_gamma1,grad_beta4,grad_beta3,grad_beta2,grad_beta1,grad_inter_x,grad_x =None, None, None, None, None, None, None, None,None,None
        grad_iter,grad_out10,grad_out9_a,grad_alpha,grad_alpha_iter,grad_out9,grad_linear3,grad_out8,grad_out7_a,grad_out5_a,grad_out3_a = None,None,None,None,None,None,None,None,None,None,  None
        weight, linear0,out10,out9_a,out9,out3,out2,out1,out1_tanh= ctx.saved_tensors
        channel,kernel,x_size,y_size,cache1= ctx.other
        grad_out10 = 2*(grad_output.flatten(start_dim=1))
        grad_out9  = grad_out10/(2*torch.max(torch.abs(out9)))
        grad_out3 = grad_out9
        grad_out2,grad_gamma1,grad_beta1 = LBNbackward(grad_out3,cache1)
        grad_out1 = grad_out2.mm(linear0.t())
        grad_linear0 = (out1.flatten(start_dim =1).t()).mm(grad_out2)
        grad_x = grad_out1
        if kernel>0 and y_size>0 and x_size>0:
            grad_x = grad_x.reshape(channel,kernel,x_size,y_size)
        else:
            grad_x = grad_x
        return grad_x, None,grad_linear0,grad_gamma1,grad_beta1,None, None,None, None,None,None,None,None,None,None


def L_STE_FN(x,weight2,linear0,gamma1,beta1,bn1,nbit,size,channel,kernel,x_size, y_size,orign_list,inter_list,MODE):
    return L_STE.apply(x,weight2,linear0,gamma1,beta1,bn1,nbit,size,channel,kernel,x_size, y_size,orign_list,inter_list,MODE)


class FC_STE(Function):
    @staticmethod
    def forward(ctx,x,weight2,linear0,gamma1,beta1,bn1,nbit,orign_list,inter_list,MODE):
        scale = 2**nbit - 1
        out1 = x
        out1_tanh=torch.tanh(x)
        out2 = out1.mm(linear0)
        out3,cache1 = FCLBNforward(out2,gamma1,beta1,bn1,MODE)
        out9 = out3
        out9_a = out9/ (2*torch.max(torch.abs(out9))) + 0.5
        out10 = out9_a
        output = 2* torch.round(out10*scale)/scale -1
        ctx.other =cache1
        orign_list[weight2] = out1.detach()
        inter_list[weight2] = out9.detach()

        ctx.save_for_backward(x, linear0,out10,out9_a,out9,out2,out1,out1_tanh,out3)
        return output
    @staticmethod
    def backward(ctx,grad_output):
        grad_out7,grad_out6,grad_out5,grad_out4,grad_out3,grad_linear2,grad_out2,grad_linear1,grad_out1,grad_linear0 = None,None,None,None,None,None,None,None,None,None
        grad_gamma4,grad_gamma3,grad_gamma2,grad_gamma1,grad_beta4,grad_beta3,grad_beta2,grad_beta1,grad_inter_x,grad_x =None, None, None, None, None, None, None, None,None,None
        grad_iter,grad_out10,grad_out9_a,grad_alpha,grad_alpha_iter,grad_out9,grad_linear3,grad_out8,grad_out7_a,grad_out5_a,grad_out3_a = None,None,None,None,None,None,None,None,None,None,  None
        weight, linear0,out10,out9_a,out9,out2,out1,out1_tanh,out3= ctx.saved_tensors
        cache1= ctx.other
        grad_out10 = 2*(grad_output)
        grad_out9  = grad_out10/(2*torch.max(torch.abs(out9)))
        grad_out3 = grad_out9
        grad_out2,grad_gamma1,grad_beta1 = FCLBNbackward(grad_out3,cache1)
        grad_out1 = grad_out2.mm(linear0.t())
        grad_linear0 = (out1.t()).mm(grad_out2)
        grad_x = grad_out1
        return grad_x,None,grad_linear0,grad_gamma1,grad_beta1,None,None,None,None,None


def FC_STE_FN(x,weight2,linear0,gamma1,beta1,bn1,nbit,orign_list,inter_list,MODE):
    return FC_STE.apply(x,weight2,linear0,gamma1,beta1,bn1,nbit,orign_list,inter_list,MODE)

#######################network_module##########################


class Conv2d(nn.Conv2d):
    def __init__(self, linear,gamma,beta,cita,bn, origin_idx,inchannels, outchannels,kernel_size, stride=1, padding=0, nbit_w=32, nbit_a=32,dilation=1,groups=1,bias = False, padding_mode = "zeros"):
        super(Conv2d, self).__init__( inchannels, outchannels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        self.quan_a = quan_a_fn
        self.L_STE = L_STE_FN
        self.kernel_size = kernel_size
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.linear = linear
        self.lbn1 = bn
        self.gamma1 = gamma
        self.beta1 = beta
        self.cita = cita
        self.origin_idx = origin_idx
        # self.orign_list = orign_list
        # self.weight_list = weight_list
        # self.inter_list = inter_list
    def forward(self, input,MODE,weight_list,orign_list,inter_list,):

        if self.nbit_w < 32:
            input_size, input_channel, input_kernel, input_x_size, input_y_size = cpt_size(self.weight)
            self.w= self.L_STE(self.weight, self.origin_idx,self.linear,self.gamma1,self.beta1,self.lbn1,self.nbit_w, input_size, input_channel, input_kernel, input_x_size, input_y_size,orign_list,inter_list,MODE)
        else:
            self.w = self.weight
        if self.nbit_a < 32:
            self.acti = self.quan_a(input, self.cita,self.nbit_a)
        else:
            self.acti = input
        output = F.conv2d(self.acti, self.w, stride = self.stride, padding = self.padding, dilation=self.dilation, groups = self.groups)
        # print("idx,",self.origin_idx)
        weight_list[self.origin_idx]=self.w.detach()

        return output

# class fConv2d(nn.Conv2d):
#     def __init__(self, cita,origin_idx,inchannels, outchannels,kernel_size, stride=1, padding=0, nbit_w=32, nbit_a=32,dilation=1,groups=1,bias = False, padding_mode = "zeros"):
#         super(fConv2d, self).__init__( inchannels, outchannels, kernel_size, stride, padding, dilation,
#             groups, bias)
#         self.nbit_w = nbit_w
#         self.nbit_a = nbit_a
#         self.quan_a = quan_a_fn
#         # self.L_STE = L_STE_FN
#         self.ste =  Ste_quanti
#         self.kernel_size = kernel_size
#         self.inchannels = inchannels
#         self.outchannels = outchannels
#         # self.linear = linear
#         # self.lbn1 = dict_set()
#         # self.gamma1 = gamma
#         # self.beta1 = beta
#         self.cita = cita
#         self.origin_idx = origin_idx
#     def forward(self, input):

#         if self.nbit_w <= 32:
#             input_size, input_channel, input_kernel, input_x_size, input_y_size = cpt_size(self.weight)
#             # print(self.nbit_w)
#             self.w = self.ste(self.weight,self.nbit_w)
#             # self.w= self.L_STE(self.weight, self.origin_idx,self.linear,self.gamma1,self.beta1,self.lbn1,self.nbit_w, input_size, input_channel, input_kernel, input_x_size, input_y_size)
#         else:
#             self.w = self.weight
#         if self.nbit_a < 32:
#             self.acti = self.quan_a(input, self.cita,self.nbit_a)
#         else:
#             self.acti = input
#         output = F.conv2d(self.acti, self.w, stride = self.stride, padding = self.padding, dilation=self.dilation, groups = self.groups)
#         weight_list[self.origin_idx]=self.w.detach()
#         return output

class fConv2d(nn.Conv2d):
    def __init__(self, linear,gamma,beta,cita,bn ,origin_idx,inchannels, outchannels,kernel_size, stride=1, padding=0, nbit_w=32, nbit_a=32,dilation=1,groups=1,bias = False, padding_mode = "zeros"):
        super(fConv2d, self).__init__( inchannels, outchannels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        self.quan_a = quan_a_fn
        self.ste =  Ste_quanti
        self.kernel_size = kernel_size
        self.inchannels = inchannels
        self.outchannels = outchannels
        self.linear = linear
        self.lbn1 = bn
        # self.gamma1 = gamma
        # self.beta1 = beta
        self.cita = cita
        self.origin_idx = origin_idx
        # self.orign_list = orign_list
        # self.weight_list = weight_list
        # self.inter_list = inter_list
    def forward(self, input,MODE,weight_list,orign_list,inter_list,):

        if self.nbit_w <= 32:
            input_size, input_channel, input_kernel, input_x_size, input_y_size = cpt_size(self.weight)
            # print(self.nbit_w)
            self.w = self.ste(self.weight,self.nbit_w)
            # self.w= self.L_STE(self.weight, self.origin_idx,self.linear,self.gamma1,self.beta1,self.lbn1,self.nbit_w, input_size, input_channel, input_kernel, input_x_size, input_y_size)
        else:
            self.w = self.weight
        if self.nbit_a < 32:
            self.acti = self.quan_a(input, self.cita,self.nbit_a)
        else:
            self.acti = input
        output = F.conv2d(self.acti, self.w, stride = self.stride, padding = self.padding, dilation=self.dilation, groups = self.groups)
        weight_list[self.origin_idx]=self.w.detach()
        return output

    
class L_LINEAR(nn.Linear):
    def __init__(self, in_features, out_features,linear,gamma,beta,cita,origin_idx,nbit_a ,nbit_w,bias = True):
        super(L_LINEAR, self).__init__(in_features, out_features, bias)
        self.in_features = in_features
        self.out_features = out_features
        self.nbit_w = nbit_w
        self.nbit_a = nbit_a
        self.FC_STE = FC_STE_FN
        self.lbn1 = dict_set()
        self.linear = linear
        self.gamma1 = gamma
        self.beta1 = beta
        self.quan_a = quan_a_fn
        self.cita = cita
        self.origin_idx = origin_idx

    def forward(self,input):
        if self.nbit_w < 32:
            self.w = self.FC_STE(self.weight,self.origin_idx,self.linear,self.gamma1,self.beta1,self.lbn1,self.nbit_w)
        else:
            self.w = self.weight
        #

        if self.nbit_a < 32:
            self.acti = self.quan_a(input, self.cita,self.nbit_a)

        else:
            self.acti = input

        output = F.linear(self.acti,self.w, self.bias)
        weight_list[self.origin_idx] = self.w.detach()
        return output
