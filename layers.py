import torch



def mm_scaling_score(gamma, beta):
    def func(score, mask_min=False):
        if mask_min:
            min_mask = torch.zeros_like(score).type(torch.int8)
            min_mask.scatter_(-1, score.min(dim=-1).indices.unsqueeze(-1), 1)
            score_temp = score + min_mask*score.max()
        else:
            score_temp = score
        
        score = ((score_temp-score_temp.min(dim=-1, keepdim=True)[0])/(score.max(dim=-1, keepdim=True)[0]-score_temp.min(dim=-1, keepdim=True)[0]+1e-6)*gamma)**beta
        score = torch.exp(0-score)
        score = score * (1-min_mask) if mask_min else score

        return score
    return func



def score_calculate(memory_bank, input_data, build_bank=False, f_score=mm_scaling_score(10, 2), norm=False, **model_args):
    # mb: [B_tr, L], id: [B_te, L]
    B_tr, L = memory_bank.shape

    if norm:
        memory_bank = memory_bank - memory_bank.mean(dim=1, keepdim=True)
        input_data = input_data - input_data.mean(dim=1, keepdim=True)
        
    score = abs(input_data.unsqueeze(dim=1).expand(-1, B_tr, -1) - memory_bank)  # [B_te, B_tr, L]
    score = score.norm(dim=-1, p=2)  # [B_te, B_tr]
    
    score = f_score(score, build_bank)
    
    score = score / score.norm(dim=-1, p=1, keepdim=True)

    return score


def decouple_layer(X_tr, Y_tr, X_data, Y_data=None, build_bank=False, batch_size=-1, gamma=10, beta=2, **model_args):
    # X_data: [B_te, L], X_tr: [B_tr, L]

    assert (Y_data is None and build_bank is False) or (Y_data is not None and build_bank is True)

    B_te, L = X_data.shape

    batch_size = B_te if batch_size == -1 else batch_size

    X_res = []
    Y_pred, Y_res = [] if not build_bank else None, [] if build_bank else None

    for i in range(0, B_te, batch_size):

        X_data_i = X_data[i:i+batch_size, :]
        
        score = score_calculate(X_tr, X_data_i, build_bank, mm_scaling_score(gamma, beta), True, nbatch_layer=i, **model_args)
        alpha = score.unsqueeze(-1).expand(-1, -1, L)


        X_res_i = X_data_i - (alpha*(X_tr - X_tr.mean(dim=1, keepdim=True))).sum(dim=1) - X_data_i.mean(dim=1, keepdim=True)
        X_res.append(X_res_i)
        
        if build_bank:
            Y_res_i = Y_data[i:i+batch_size, :] - (alpha*(Y_tr - X_tr.mean(dim=1, keepdim=True))).sum(dim=1) - X_data_i.mean(dim=1, keepdim=True)
            Y_res.append(Y_res_i)
        else:
            Y_pred_i = (alpha*(Y_tr - X_tr.mean(dim=1, keepdim=True))).sum(dim=1) + X_data_i.mean(dim=1, keepdim=True)
            Y_pred.append(Y_pred_i)



    X_res = torch.cat(X_res)
    Y_pred, Y_res = torch.cat(Y_pred) if not build_bank else None, torch.cat(Y_res) if build_bank else None

    return Y_pred, X_res, Y_res

def timewise_decouple_layer(X_tr, Y_tr, X_data, Y_data=None, tid_dif=0, tid_count=288, tid_tol=3, build_bank=False, gamma=10, beta=2, **model_args):
    # X_data: [B_te, L], X_tr: [B_tr, L]

    assert (Y_data is None and build_bank is False) or (Y_data is not None and build_bank is True)

    B_te, L = X_data.shape
    B_tr, L_Y = Y_tr.shape

    Y_pred = torch.empty(B_te, L_Y).type(X_data.type()) if not build_bank else None
    X_res = torch.empty_like(X_data)
    Y_res = torch.empty(B_te, L_Y).type(X_data.type()) if build_bank else None


    for t in range(tid_count):

        t_tr = (t + tid_dif)%tid_count

        X_data_t = X_data[t::tid_count]
        Y_data_t = Y_data[t::tid_count] if build_bank else None

        X_tr_t = torch.cat([X_tr[(t_+tid_count)%tid_count::tid_count] for t_ in range(t_tr-tid_tol, t_tr+tid_tol+1)])
        Y_tr_t = torch.cat([Y_tr[(t_+tid_count)%tid_count::tid_count] for t_ in range(t_tr-tid_tol, t_tr+tid_tol+1)])

        

        score = score_calculate(X_tr_t, X_data_t, build_bank, mm_scaling_score(gamma, beta), False, nbatch_layer=t, **model_args)
        alpha = score.unsqueeze(-1).expand(-1, -1, L)



        X_res_t = X_data_t - (alpha*X_tr_t).sum(dim=1)
        X_res[t::tid_count] = X_res_t

        if build_bank:
            Y_res_t = Y_data_t - (alpha*Y_tr_t).sum(dim=1)
            Y_res[t::tid_count] = Y_res_t
        else:
            Y_pred_t = (alpha*Y_tr_t).sum(dim=1)
            Y_pred[t::tid_count] = Y_pred_t

    return Y_pred, X_res, Y_res
