from hylearn.lib.modules.sample_backbone import shallow_net
from hylearn.lib.modules.batch import dict_set
import torch
def dpcreate(channel):
    linear_list = []
    flinear_list = []
    old_list = []
    gamma_list = []
    beta_list = []
    alpha_list = []
    cita_list = []
    weight_list = []
    orign_list = []
    inter_list = []
    bn_list = []
    sample_model = shallow_net(channel)
    for name, parameters in sample_model.named_parameters():
        if 'weight' in name:
            if 'conv' in name:
                weight1 = torch.zeros(parameters.size(0), parameters.size(1), parameters.size(2),
                                      parameters.size(3))
                weight_list.append(weight1)
                weight2 = torch.zeros(parameters.size(0), parameters.size(1), parameters.size(2),
                                      parameters.size(3))
                weight3 = torch.zeros(parameters.size(0), parameters.size(1), parameters.size(2),
                                      parameters.size(3))
                orign_list.append(weight2)
                inter_list.append(weight3)
                # if 'layer' not in name:
                # linear1 = torch.empty(parameters.size(1)*parameters.size(2)*parameters.size(3),parameters.size(1)*parameters.size(2)*parameters.size(3)).to(device)
                linear1 = (torch.eye(parameters.size(1) * parameters.size(2) * parameters.size(3),
                                     parameters.size(1) * parameters.size(2) * parameters.size(
                                         3)) + 1e-7 * torch.ones(
                    parameters.size(1) * parameters.size(2) * parameters.size(3),
                    parameters.size(1) * parameters.size(2) * parameters.size(3))).to('cuda')
                gamma1 = torch.ones([parameters.size(0), 1]).to('cuda')
                beta1 = 1e-5 * torch.ones([parameters.size(0), 1]).to('cuda')

                bn1 = dict_set(
                    torch.zeros(parameters.size(0), parameters.size(1) * parameters.size(2) * parameters.size(3)),
                    torch.zeros(parameters.size(0), parameters.size(1) * parameters.size(2) * parameters.size(3)))

                linear1.requires_grad = True
                linear1.retain_grad()
                linear_list.append(linear1)
                gamma1.requires_grad = True
                gamma1.retain_grad()
                beta1.requires_grad = True
                beta1.retain_grad()
                # bn1.requires_grad = True
                # bn1.retain_grad()
                gamma_list.append(gamma1)
                beta_list.append(beta1)
                cita = torch.Tensor([30]).to('cuda')
                cita.requires_grad = True
                cita.retain_grad()
                cita_list.append(cita)
                bn_list.append(bn1)
                # print(len(cita_list))
    cita = torch.Tensor([30]).to('cuda')
    cita.requires_grad = True
    cita.retain_grad()
    cita_list.append(cita)
    gamma1 = torch.ones(1).to('cuda')
    beta1 = 1e-5 * torch.ones(1).to('cuda')
    linear1 = (torch.eye(sample_model.out_dim,
                         sample_model.out_dim) + 1e-7 * torch.ones(
        sample_model.out_dim,
        sample_model.out_dim)).to('cuda')
    weight2 = torch.zeros(sample_model.out_dim,
                          sample_model.out_dim)
    weight3 = torch.zeros(sample_model.out_dim,
                          sample_model.out_dim)
    orign_list.append(weight2)
    inter_list.append(weight3)
    linear1.requires_grad = True
    linear1.retain_grad()
    linear_list.append(linear1)
    gamma1.requires_grad = True
    gamma1.retain_grad()
    beta1.requires_grad = True
    beta1.retain_grad()
    gamma_list.append(gamma1)
    beta_list.append(beta1)
    bn1 = dict_set(torch.zeros(1), torch.zeros(1))
    bn_list.append(bn1)
    return [linear_list,gamma_list,beta_list,cita_list,bn_list,weight_list,orign_list,inter_list]