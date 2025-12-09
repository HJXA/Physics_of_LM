# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Author: Zeyuan Allen-Zhu
#
from __future__ import print_function, absolute_import

import torch.nn as nn
import torch
import json
import os
import random
import collections
import numpy as np
import xlsxwriter
import itertools
from tqdm import tqdm


class CFG_Node():
    rng = None

    def __init__(self, config, depth, id, name):
        self.config = config
        self.depth = depth
        self.id = id
        self.name = name
        self.children = None

    def generate_leaf(self):
        if hasattr(self.config,'content_depth'):
            # T symbol ids start from self.vocab_size+4
            return [self.config.content_map[self.id-3-self.config.vocab_size-1][CFG_Config.content_chosen], self.id]
        if hasattr(self.config,'multi_vocab'):
            # real T symbol ids start from self.vocab_size+4
            choice = self.config.content_map[self.id-3-self.config.vocab_size-1]
            if 'NU1' in self.config.multi_vocab:
                return [random.choices(choice, range(1,len(choice)+1), k=1)[0], self.id]
            else:
                return [choice[random.randint(0, len(choice)-1)], self.id]
        return [self.id]


    def generate(self, parents=[]):
        random = CFG_Node.rng
        if self.depth==0:
            CFG_Node.nt_counter = [1]*15
        CFG_Node.nt_counter[self.depth] += 1
        if self.children is None: 
            return [self.generate_leaf() + [-CFG_Node.nt_counter[self.depth]] + parents]
        cc = random.randint(0,len(self.children)-1)
        res = []
        for c in self.children[cc]:
            res += self.config.all[self.depth+1][c].generate([self.id,-CFG_Node.nt_counter[self.depth]] + parents)
        return res



class CFG_Config():
    def __init__(self, depth=7, num_sym=30, deg_min=2, deg_max=3, len_min=1, len_max=2, 
                 disallow_duplicate_sym = False, disallow_duplicate_seq = False, 
                 double_data_layer = None,
                 content_depth = None, content_count = None, content_num_sym = None, multi_vocab = None, add_len_one = False):
        self.ptb = None
        self.add_len_one = add_len_one
        self.depth = depth
        self.num_sym = num_sym # number of T symbols
        self.vocab_size = self.num_sym  # may be different if using content based
        # by default terminal (vocab) symbols = 1,2,...self.num_sym
        # by default NT symbols starts from self.num_sym+4 --- skipping 3 symbols such as <EOS>
        self.deg_min = deg_min
        self.deg_max = deg_max
        self.len_min = len_min
        self.len_max = len_max
        self.num_sym_mode = 1
        self.disallow_duplicate_sym = disallow_duplicate_sym   #before Jan/26/2023 this means self.num_sym_mode>=2 
        self.disallow_duplicate_seq = disallow_duplicate_seq    #before Jan/26/2023 this means self.num_sym_mode>=2
        if content_depth is not None:
            # content (vocab) symbols = 1,2,3,...,self.vocab_size
            # terminal symbols = self.vocab_size+3+(1~self.num_sym)
            self.content_depth = content_depth
            self.content_count = content_count
            self.vocab_size = content_num_sym
            assert content_count is not None
            assert content_num_sym is not None
        elif multi_vocab is not None:
            self.multi_vocab = multi_vocab
            self.vocab_size = content_num_sym
            assert content_num_sym is not None
            assert content_count is None
        else:
            assert content_count is None
            assert content_num_sym is None
        self.eos_token = self.vocab_size+1
        self.mask_token = self.vocab_size+2
        if double_data_layer is not None:
            self.double_data_layer = double_data_layer
        self.sep_token = self.vocab_size+3

    @staticmethod
    def from_graph(file):
        bla = CFG_Config()
        delattr(bla, 'vocab_size')
        delattr(bla, 'sep_token')
        bla.read_graph(file)
        if not hasattr(bla, 'vocab_size'): # to make it compatible with earlier format
            bla.vocab_size = bla.num_sym
        if not hasattr(bla, 'sep_token'): # to make it compatible with earlier format
            bla.sep_token = bla.vocab_size + 3
        return bla

    def read_graph(self, file):
        with open(file, 'r') as openfile:
            dd = json.load(openfile)
        for key in dd.keys():
            if key!='all':
                if dd[key]=='true':
                    setattr(self,key,True)
                elif dd[key]=='false':
                    setattr(self,key,False)
                else:
                    setattr(self,key,dd[key])
        self.all = [[None for i in range(self.sizes[p])] for p in range(self.depth+1)] 
        for i in range(self.depth, -1, -1):
            for j in range(self.sizes[i]):
                self.all[i][j] = CFG_Node(self, depth=i, id=self.idx[i][j], name=dd['all'][i][j]['name'])
                if i!=self.depth:
                    self.all[i][j].children = dd['all'][i][j]['children']
        if not hasattr(self,'vocab_size'):   # in earlier versions we didn't have .vocab_size
            self.vocab_size = self.num_sym

    def print_graph(self, file = None, print_to_screen = True):
        if file:
            file = open(file,'w')
        if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
            for i in range(self.depth):
                for j in range(self.sizes[i]):
                    for child in self.all[i][j].children:
                        if file:
                            for _ in range(i): file.write("\t")
                            file.write(str(self.all[i][j].id) + "|->")
                            for c in child:
                                file.write(str(self.all[i+1][c].id) + " ")
                            file.write('\n')
                        if print_to_screen:
                            for _ in range(i): print("\t",end='')
                            print(str(self.all[i][j].id) + "|->",end='')
                            for c in child:
                                print(str(self.all[i+1][c].id) + " ",end='')
                            print()
            if file:
                file.write(str(vars(self)))
                file.write('\n')
                if hasattr(self,'multi_vocab'):
                    for a in self.content_map:
                        file.write(str(a))
                        file.write('\n')
                file.close()
            if print_to_screen:
                print(vars(self))
                if hasattr(self,'multi_vocab'):
                    for a in self.content_map:
                        print(a)

    def build_graph(self):
        self.sizes = [0] * (self.depth+1)
        self.sizes[0] = 1
        if isinstance(self.num_sym_mode, list):
            assert len(self.num_sym_mode) == self.depth+1
            self.sizes = self.num_sym_mode
        elif self.num_sym_mode in [1,3]:
            for i in range(1,self.depth+1): self.sizes[i] = self.num_sym * i // self.depth
        elif self.num_sym_mode==2:
            for i in range(1,self.depth+1): self.sizes[i] = self.num_sym

        # if content based, num_sym may be much larger than # of T of last layer
        assert self.sizes[-1]==self.num_sym

        self.idx = [[None for i in range(p)] for p in self.sizes]
        names = [[None for i in range(p)] for p in self.sizes]
        self.all = [[None for i in range(p)] for p in self.sizes]
        
        count = 0
        if hasattr(self, 'content_depth') or hasattr(self, 'multi_vocab'):
            count = self.vocab_size+3   # skip three numbers, for <SEP> <MASK> and future use
            self.sizes += [self.vocab_size]
        print("sizes: " + str(self.sizes))
        for i in range(self.depth, -1, -1):
            for j in range(self.sizes[i]):
                count += 1
                self.idx[i][j] = count
                names[i][j] = xlsxwriter.utility.xl_col_to_name(count+25)
            if i==self.depth and (not hasattr(self, 'content_depth')) and (not hasattr(self, 'multi_vocab')): 
                # skip three numbers, for <SEP> <MASK> and future use
                count += 3 
        print("names: " + str(names))
        assert count == sum(self.sizes)+3  
        self.count = count

        for i in range(self.depth, -1, -1):
            print(i)
            depth_hash = []
            for j in range(self.sizes[i]):
                self.all[i][j] = CFG_Node(self, depth=i, id=self.idx[i][j], name=names[i][j])
                cur = self.all[i][j]
                if i!=self.depth:
                    degree = random.randint(self.deg_min, self.deg_max)
                    if self.add_len_one and i!=0 and j==0: degree=1
                    cur.children = []
                    for k in range(degree):
                        llen = random.randint(self.len_min, self.len_max)
                        if self.add_len_one and i!=0 and j==0 and k==0: llen = 1
                        # if llen==1:  # 降低len=1的出现概率，否则solution太不唯一
                        #     llen = random.randint(self.len_min, self.len_max)
                        child = []
                        child_n = ""
                        for l in range(llen):
                            nextid = random.randint(0,self.sizes[i+1]-1)
                            if self.add_len_one and i!=0 and j==0 and k==0: nextid = 0
                            # if llen==1 and l==0:
                            #     nextid = random.randint(j,self.sizes[i+1]-1)
                            while self.disallow_duplicate_sym and nextid in child:
                                nextid = random.randint(0,self.sizes[i+1]-1)
                            child += [nextid]
                            child_n += "|" + str(nextid)
                        while self.disallow_duplicate_seq and child_n in depth_hash:
                            llen = random.randint(self.len_min, self.len_max)
                            # if llen==1:  # 降低len=1的出现概率，否则solution太不唯一
                            #     llen = random.randint(self.len_min, self.len_max)
                            child = []
                            child_n = ""
                            for l in range(llen):
                                nextid = random.randint(0,self.sizes[i+1]-1)
                                # if llen==1 and l==0:
                                #     nextid = random.randint(j,self.sizes[i+1]-1)
                                #     print(nextid, depth_hash)
                                while self.disallow_duplicate_sym and nextid in child:
                                    nextid = random.randint(0,self.sizes[i+1]-1)
                                child += [nextid]
                                child_n += "|" + str(nextid)
                        cur.children += [child]
                        depth_hash += [child_n]

        if hasattr(self,'content_depth'):
            # sizes[-1] x content_count -> 
            self.content_map = [ [random.randint(1, self.vocab_size) for _ in range(self.content_count)] for _ in range(self.num_sym) ]
        if hasattr(self,'multi_vocab'):
            if 'disjoint' in self.multi_vocab:
                self.content_map = [ [] for _ in range(self.sizes[-2]) ]
                for i in range(self.vocab_size):
                    self.content_map[i%len(self.content_map)] += [i+1]
            elif 'overlap' in self.multi_vocab:
                self.content_map = [ ]
                for i in range(self.sizes[-2]):
                    self.content_map += [  [random.randint(1, self.vocab_size) for _ in range(self.vocab_size * 3 // (2 * self.sizes[-2]) )]  ]
                    self.content_map[-1] = sorted(set(self.content_map[-1]))
        #self.print_graph()

    def save_graph(self, file):
        dd = vars(self)
        for i in range(len(self.all)):
            for j in range(len(self.all[i])):
                dd['all'][i][j] = vars(self.all[i][j])
                del dd['all'][i][j]['config']
        with open(file, "w") as outfile:
            json.dump(dd, outfile)

    def generate_onedata(self, rng):
        if self.ptb is not None:
            sentence = self.ptb.generate_sentence(rng)
            b = self.ptb.sentence_to_int(sentence)
            return [[a] for a in b]
        random = rng
        CFG_Node.rng = rng
        if hasattr(self, 'double_data_layer'):
            assert False, "code reserved for other non-published use"
        if hasattr(self,'content_depth'):
            assert False, "code reserved for other non-published use"

        tmp_rand_flip = None
        if hasattr(self, 'rand_flip') and isinstance(self.rand_flip, list):
            if random.uniform(0,1)<=float(self.rand_flip[0]):  # with probability "self.rand_flip[0]" we do not randomly flip
                tmp_rand_flip = self.rand_flip
                delattr(self, 'rand_flip')
        output = self.all[0][0].generate()
        if tmp_rand_flip is not None:
            self.rand_flip = tmp_rand_flip

        return output

    def generate_onedata_pure(self, rng):
        bla = self.generate_onedata(rng)
        return [a[0] for a in bla]
        

    # 这是solve_dp_noneq的快速版，只计算是否能satisfy CFG（不计算需要最少修改多少个token才能satisfy），并且如果能satisfy，会输出有多少种可能satisfy CFG
    def solve_dp_noneq_fast(self, aa, no_debug=False, unique_path=None, cont = False, 
                       cont_sym = None, full_sol = True, kill_front = None, 
                       cascade_count = True, unknown_root = False,
                       double_seq = None):
        if self.ptb is not None:
            c = self.ptb.is_in_cfg(aa)
            return c, None, None, None
        ##print(f"begin solve_dp_noneq {hash(tuple(aa))}: {aa}")
        ll = len(aa)
        if not cont:
            opt = [ [ [ [10000] for _ in range(self.count+1) ] for _ in range(ll)] for _ in range(ll)]
            opt_c = [ [ [ [0] for _ in range(self.count+1) ] for _ in range(ll)] for _ in range(ll)]
            ffrom = [ [ [ [None] for _ in range(self.count+1) ] for _ in range(ll)] for _ in range(ll)]
            look_left = [ [ [] for _ in range(self.count+1) ] for _ in range(ll)] 
            id_to_ab = np.full((self.count+1, 2), 0)

            for k in range(ll):
                if hasattr(self,'multi_vocab'):
                    # real T symbol ids start from self.vocab_size+4
                    for ec in range(self.sizes[self.depth]):
                        choice = self.content_map[self.all[self.depth][ec].id-3-self.vocab_size-1]
                        if aa[k] in choice:
                            opt[k][k][ self.all[self.depth][ec].id ][0] = 0
                            opt_c[k][k][ self.all[self.depth][ec].id ][0] = 1
                            if (k-1) not in look_left[k][self.all[self.depth][ec].id]:
                                look_left[k][self.all[self.depth][ec].id] += [k-1]
                else:
                    if aa[k] in self.idx[self.depth]:
                        opt[k][k][aa[k]][0] = 0
                        opt_c[k][k][aa[k]][0] = 1
                        if (k-1) not in look_left[k][aa[k]]:
                            look_left[k][aa[k]] += [k-1]
            for a in range(self.depth+1):
                for b in range(self.sizes[a]):
                    cur = self.all[a][b].id
                    id_to_ab[cur,0] = a
                    id_to_ab[cur,1] = b
            for k in range(0, ll):
                if not no_debug: print('.', end='', flush=True)
                ##print(f"during solve_dp_noneq {hash(tuple(aa))}:{k}")
                for i in range(0,ll-k):
                    j = i+k
                    for a in reversed(range(self.depth)):
                        for b in range(self.sizes[a]):
                            cur = self.all[a][b].id
                            opt_pos = 0
                            for child in self.all[a][b].children:
                                if len(child)==1:
                                    x = child[0]
                                    x = self.all[a+1][x].id
                                    if opt[i][j][x][0] < opt[i][j][cur][0]:
                                        if opt[i][j][cur][0]==10000:
                                            look_left[j][cur] += [i-1]
                                        opt[i][j][cur][0] = opt[i][j][x][0]          # 最少editing dis
                                        opt_c[i][j][cur][0] = opt_c[i][j][x][0]      # sol个数
                                        ffrom[i][j][cur][0] = ['single', x, None]
                                        #print('wa')
                                    elif opt[i][j][x][0] == opt[i][j][cur][0]:
                                        opt_c[i][j][cur][0] += opt_c[i][j][x][0]
                                elif len(child)==2:
                                    x = child[0]
                                    y = child[1]
                                    x = self.all[a+1][x].id
                                    y = self.all[a+1][y].id
                                    #for mid in range(i, j):
                                    for mid in look_left[j][y]:
                                        if mid<i: break;
                                        if opt[i][mid][x][0] + opt[mid+1][j][y][0] < opt[i][j][cur][0]:
                                            if opt[i][j][cur][0]==10000:
                                                look_left[j][cur] += [i-1]
                                            opt[i][j][cur][0] = opt[i][mid][x][0] + opt[mid+1][j][y][0]
                                            opt_c[i][j][cur][0] = opt_c[i][mid][x][0] * opt_c[mid+1][j][y][0]
                                            ffrom[i][j][cur][0] = [mid, x, y]
                                        elif opt[i][mid][x][0] + opt[mid+1][j][y][0] == opt[i][j][cur][0]:
                                            opt_c[i][j][cur][0] += opt_c[i][mid][x][0] + opt_c[mid+1][j][y][0]
                                else:
                                    for i_c in range(len(child)-2):
                                        opt_pos+=1
                                        assert len(opt[i][j][cur])==opt_pos
                                        assert len(ffrom[i][j][cur])==opt_pos
                                        opt[i][j][cur] += [10000]
                                        opt_c[i][j][cur] += [0]
                                        ffrom[i][j][cur] += [None]
                                        x = child[i_c]
                                        y = child[i_c+1]
                                        x = self.all[a+1][x].id
                                        y = self.all[a+1][y].id
                                        #for mid in range(i, j):
                                        for mid in look_left[j][y]:
                                            if mid<i: break;
                                            if i_c==0: 
                                                left_opt = opt[i][mid][x][0]
                                                left_opt_c = opt_c[i][mid][x][0]
                                            else: 
                                                left_opt = opt[i][mid][cur][opt_pos-1]
                                                left_opt_c = opt_c[i][mid][cur][opt_pos-1]
                                                assert False, "CFG node length <= 3; haven't yet implemented (我太忙了)"
                                            if left_opt + opt[mid+1][j][y][0] < opt[i][j][cur][opt_pos]:
                                                opt[i][j][cur][opt_pos] = left_opt + opt[mid+1][j][y][0]
                                                opt_c[i][j][cur][opt_pos] = left_opt_c * opt_c[mid+1][j][y][0]
                                                ffrom[i][j][cur][opt_pos] = [mid, x, y]
                                            elif left_opt + opt[mid+1][j][y][0] == opt[i][j][cur][opt_pos]:
                                                opt_c[i][j][cur][opt_pos] += left_opt_c * opt_c[mid+1][j][y][0]
                                    x = child[-2]
                                    y = child[-1]
                                    x = self.all[a+1][x].id
                                    y = self.all[a+1][y].id
                                     #for mid in range(i, j):
                                    for mid in look_left[j][y]:
                                        if mid<i: break;
                                        if len(opt[i][mid][cur])<=opt_pos: continue
                                        left_opt = opt[i][mid][cur][opt_pos]
                                        left_opt_c = opt_c[i][mid][cur][opt_pos]
                                        if left_opt + opt[mid+1][j][y][0] < opt[i][j][cur][0]:
                                            if opt[i][j][cur][0]==10000:
                                                look_left[j][cur] += [i-1]
                                            opt[i][j][cur][0] = left_opt + opt[mid+1][j][y][0]
                                            opt_c[i][j][cur][0] = left_opt_c * opt_c[mid+1][j][y][0]
                                            ffrom[i][j][cur][0] = [mid, -opt_pos, y]
                                        elif left_opt + opt[mid+1][j][y][0] == opt[i][j][cur][0]:
                                            opt_c[i][j][cur][0] += left_opt_c * opt_c[mid+1][j][y][0]
            if not no_debug: print('')
            self.opt = opt
            self.opt_c = opt_c
            self.ffrom = ffrom
            self.id_to_ab = id_to_ab
            ## print(f"finished dp {hash(tuple(aa))}")
            # print(self.opt[2][2][2])
            # print(self.opt[2][2][7])
            # print(self.opt[0][1][8])
            # print(self.opt[0][2][10])
            # print(self.opt[0][9][14])
            # print(self.opt[0][21][16])
            # print(self.opt[0][51][21])
            # print(self.opt[0][122][22])
        else:
            assert False, "code reserved for other non-published use"


        # counts计算每层错的NT个数
        # opt_c计算总的opt=opt的可能性（用于推算二义性）
        # tot_sol[a] 代表第a个token从root往下的编号
        # tot_sol2[a]代表第a个token所对应NT/T是第几个（的负数），和cfg data的格式类似
        def calc_dp_res(i,j,p,q, stop_layer=None):
            if j<0: # in the event that ll=0
                return [None], None, None
            tot_sol1 = []
            tot_sol2 = []
            counts = [0 for _ in range(self.depth+1)]
            cal_order = [1 for _ in range(self.depth+1)]
            def myprint(i,j,cur,spx,lst1,lst2):
                nonlocal tot_sol1, tot_sol2, counts
                if not no_debug:
                    print(i,j,cur,spx,lst1,lst2)
                if spx==0:
                    cal_order[len(lst1)]+=1
                    lst2 = lst2 + [-cal_order[len(lst1)]]
                    lst1 = lst1 + [cur]
                    if opt[i][j][cur][spx]!=0:
                        counts[self.depth+1 - len(lst1)]+=1
                if (i==j and ffrom[i][j][cur][spx] is None) or len(lst1)==stop_layer:
                    tot_sol1 += [lst1]
                    tot_sol2 += [lst2]
                else:
                    fffrom = ffrom[i][j][cur][spx]
                    if fffrom is None: return
                    mid = fffrom[0]
                    if mid=='single':
                        x = fffrom[1]
                        myprint(i,j,x,0,lst1,lst2)
                    elif fffrom[1]>=0:
                        x = fffrom[1]
                        y = fffrom[2]
                        myprint(i,mid,x,0,lst1,lst2)
                        myprint(mid+1,j,y,0,lst1,lst2)
                    else:
                        spx_n = -fffrom[1]
                        y = fffrom[2]
                        myprint(i,mid,cur,spx_n,lst1,lst2)
                        myprint(mid+1,j,y,0,lst1,lst2)
            if opt[i][j][p][q]>9999:
                return [None], None, None
            myprint(i,j,p,q,[],[])
            tot_sol = []
            for sol1, sol2 in zip(tot_sol1, tot_sol2):
                tot_sol += [ [val for pair in zip(sol2, sol1) for val in pair][::-1] ]
            assert tot_sol[-1][1] == -len(tot_sol)-1 and tot_sol[-1][-1]==-2, f"你是不是傻，最后一行的输出是{tot_sol[-1]}"

            return tot_sol, counts, opt_c[i][j][p][q]

        # sequence重复两次，选择最佳cut点
        # 两个sequence都stop_layer = double_seq，也就是只生成到NT所在层        
        if double_seq is not None:
            assert False, "code reserved for other non-published use"
            stop_layer = abs(double_seq)+1

        if unique_path is not None:
            assert False, "code reserved for other non-published use"

        if no_debug and not full_sol:
            if unknown_root:
                sol = -1
                for i in range(self.count+1):
                    if opt[0][ll-1][i][0]==0:
                        if sol!=-1: print("multiple!")
                        sol = i
                return sol
            return opt[0][ll-1][self.all[0][0].id][0] #, optc[0][ll-1][self.all[0][0].id]

        if not no_debug:
            print(f"min_change = {opt[0][ll-1][self.all[0][0].id][0]}")
            #print(f"possibility = {optc[0][ll-1][self.all[0][0].id]}")

        tot_sol1, counts, num_poss = calc_dp_res(0,ll-1,self.all[0][0].id,0)
        if not no_debug:
            print(tot_sol1)

        best = 10000 if ll==0 else opt[0][ll-1][self.all[0][0].id][0]
        if full_sol:
            if cascade_count:
                return best, tot_sol1, counts, num_poss

            return best, tot_sol1

        return best #, optc[0][ll-1][self.all[0][0].id]

        #print(opt[0][0][5])
        #print(opt[1][1][14])
        #print(opt[0][1][21])




    # DP solver for Non EQ length -- CFG <> exactly binary tree 
    # cont / unique_path are not useful here
    def solve_dp_noneq(self, aa, no_debug=False, unique_path=None, cont = False, 
                       cont_sym = None, full_sol = True, kill_front = None, 
                       cascade_count = True, unknown_root = False,
                       double_seq = None):
        if self.ptb is not None:
            c = self.ptb.is_in_cfg(aa)
            return c, None, None, None
        ##print(f"begin solve_dp_noneq {hash(tuple(aa))}: {aa}")
        ll = len(aa)
        if not cont:
            opt = [ [ [ [10000] for _ in range(self.count+1) ] for _ in range(ll)] for _ in range(ll)]
            opt_c = [ [ [ [0] for _ in range(self.count+1) ] for _ in range(ll)] for _ in range(ll)]
            ffrom = [ [ [ [None] for _ in range(self.count+1) ] for _ in range(ll)] for _ in range(ll)]
            look_left = [ [ [] for _ in range(self.count+1) ] for _ in range(ll)] 
            id_to_ab = np.full((self.count+1, 2), 0)

            for k in range(ll):
                for t in range(self.sizes[self.depth]):
                    if kill_front is not None and k<kill_front:
                        pass
                    else:
                        opt[k][k][ self.all[self.depth][t].id ][0] = 1
                        opt_c[k][k][ self.all[self.depth][t].id ][0] = 1
                        look_left[k][self.all[self.depth][t].id] += [k-1]
                if hasattr(self,'multi_vocab'):
                    # real T symbol ids start from self.vocab_size+4
                    for ec in range(self.sizes[self.depth]):
                        choice = self.content_map[self.all[self.depth][ec].id-3-self.vocab_size-1]
                        if aa[k] in choice:
                            opt[k][k][ self.all[self.depth][ec].id ][0] = 0
                            opt_c[k][k][ self.all[self.depth][ec].id ][0] = 1
                            if (k-1) not in look_left[k][self.all[self.depth][ec].id]:
                                look_left[k][self.all[self.depth][ec].id] += [k-1]
                else:
                    if aa[k] in self.idx[self.depth]:
                        opt[k][k][aa[k]][0] = 0
                        opt_c[k][k][aa[k]][0] = 1
                        if (k-1) not in look_left[k][aa[k]]:
                            look_left[k][aa[k]] += [k-1]
            for a in range(self.depth+1):
                for b in range(self.sizes[a]):
                    cur = self.all[a][b].id
                    id_to_ab[cur,0] = a
                    id_to_ab[cur,1] = b
            for k in range(0, ll):
                if not no_debug: print('.', end='', flush=True)
                ##print(f"during solve_dp_noneq {hash(tuple(aa))}:{k}")
                for i in range(0,ll-k):
                    j = i+k
                    for a in reversed(range(self.depth)):
                        for b in range(self.sizes[a]):
                            cur = self.all[a][b].id
                            opt_pos = 0
                            for child in self.all[a][b].children:
                                if len(child)==1:
                                    x = child[0]
                                    x = self.all[a+1][x].id
                                    if opt[i][j][x][0] < opt[i][j][cur][0]:
                                        if opt[i][j][cur][0]==10000:
                                            look_left[j][cur] += [i-1]
                                        opt[i][j][cur][0] = opt[i][j][x][0]          # 最少editing dis
                                        opt_c[i][j][cur][0] = opt_c[i][j][x][0]      # sol个数
                                        ffrom[i][j][cur][0] = ['single', x, None]
                                        #print('wa')
                                    elif opt[i][j][x][0] == opt[i][j][cur][0]:
                                        opt_c[i][j][cur][0] += opt_c[i][j][x][0]
                                elif len(child)==2:
                                    x = child[0]
                                    y = child[1]
                                    x = self.all[a+1][x].id
                                    y = self.all[a+1][y].id
                                    #for mid in range(i, j):
                                    for mid in look_left[j][y]:
                                        if mid<i: break;
                                        if opt[i][mid][x][0] + opt[mid+1][j][y][0] < opt[i][j][cur][0]:
                                            if opt[i][j][cur][0]==10000:
                                                look_left[j][cur] += [i-1]
                                            opt[i][j][cur][0] = opt[i][mid][x][0] + opt[mid+1][j][y][0]
                                            opt_c[i][j][cur][0] = opt_c[i][mid][x][0] * opt_c[mid+1][j][y][0]
                                            ffrom[i][j][cur][0] = [mid, x, y]
                                        elif opt[i][mid][x][0] + opt[mid+1][j][y][0] == opt[i][j][cur][0]:
                                            opt_c[i][j][cur][0] += opt_c[i][mid][x][0] + opt_c[mid+1][j][y][0]
                                else:
                                    for i_c in range(len(child)-2):
                                        opt_pos+=1
                                        assert len(opt[i][j][cur])==opt_pos
                                        assert len(ffrom[i][j][cur])==opt_pos
                                        opt[i][j][cur] += [10000]
                                        opt_c[i][j][cur] += [0]
                                        ffrom[i][j][cur] += [None]
                                        x = child[i_c]
                                        y = child[i_c+1]
                                        x = self.all[a+1][x].id
                                        y = self.all[a+1][y].id
                                        #for mid in range(i, j):
                                        for mid in look_left[j][y]:
                                            if mid<i: break;
                                            if i_c==0: 
                                                left_opt = opt[i][mid][x][0]
                                                left_opt_c = opt_c[i][mid][x][0]
                                            else: 
                                                left_opt = opt[i][mid][cur][opt_pos-1]
                                                left_opt_c = opt_c[i][mid][cur][opt_pos-1]
                                                assert False, "CFG node length <= 3; haven't yet implemented (我太忙了)"
                                            if left_opt + opt[mid+1][j][y][0] < opt[i][j][cur][opt_pos]:
                                                opt[i][j][cur][opt_pos] = left_opt + opt[mid+1][j][y][0]
                                                opt_c[i][j][cur][opt_pos] = left_opt_c * opt_c[mid+1][j][y][0]
                                                ffrom[i][j][cur][opt_pos] = [mid, x, y]
                                            elif left_opt + opt[mid+1][j][y][0] == opt[i][j][cur][opt_pos]:
                                                opt_c[i][j][cur][opt_pos] += left_opt_c * opt_c[mid+1][j][y][0]
                                    x = child[-2]
                                    y = child[-1]
                                    x = self.all[a+1][x].id
                                    y = self.all[a+1][y].id
                                     #for mid in range(i, j):
                                    for mid in look_left[j][y]:
                                        if mid<i: break;
                                        if len(opt[i][mid][cur])<=opt_pos: continue
                                        left_opt = opt[i][mid][cur][opt_pos]
                                        left_opt_c = opt_c[i][mid][cur][opt_pos]
                                        if left_opt + opt[mid+1][j][y][0] < opt[i][j][cur][0]:
                                            if opt[i][j][cur][0]==10000:
                                                look_left[j][cur] += [i-1]
                                            opt[i][j][cur][0] = left_opt + opt[mid+1][j][y][0]
                                            opt_c[i][j][cur][0] = left_opt_c * opt_c[mid+1][j][y][0]
                                            ffrom[i][j][cur][0] = [mid, -opt_pos, y]
                                        elif left_opt + opt[mid+1][j][y][0] == opt[i][j][cur][0]:
                                            opt_c[i][j][cur][0] += left_opt_c * opt_c[mid+1][j][y][0]
            if not no_debug: print('')
            self.opt = opt
            self.opt_c = opt_c
            self.ffrom = ffrom
            self.id_to_ab = id_to_ab
            ## print(f"finished dp {hash(tuple(aa))}")
            # print(self.opt[2][2][2])
            # print(self.opt[2][2][7])
            # print(self.opt[0][1][8])
            # print(self.opt[0][2][10])
            # print(self.opt[0][9][14])
            # print(self.opt[0][21][16])
            # print(self.opt[0][51][21])
            # print(self.opt[0][122][22])
        else:
            assert False, "code reserved for other non-published use"

        # counts计算每层错的NT个数
        # opt_c计算总的opt=opt的可能性（用于推算二义性）
        # tot_sol[a] 代表第a个token从root往下的编号
        # tot_sol2[a]代表第a个token所对应NT/T是第几个（的负数），和cfg data的格式类似
        def calc_dp_res(i,j,p,q, stop_layer=None):
            if j<0: # in the event that ll=0
                return [None], None, None
            tot_sol1 = []
            tot_sol2 = []
            counts = [0 for _ in range(self.depth+1)]
            cal_order = [1 for _ in range(self.depth+1)]
            def myprint(i,j,cur,spx,lst1,lst2):
                nonlocal tot_sol1, tot_sol2, counts
                if not no_debug:
                    print(i,j,cur,spx,lst1,lst2)
                if spx==0:
                    cal_order[len(lst1)]+=1
                    lst2 = lst2 + [-cal_order[len(lst1)]]
                    lst1 = lst1 + [cur]
                    if opt[i][j][cur][spx]!=0:
                        counts[self.depth+1 - len(lst1)]+=1
                if (i==j and ffrom[i][j][cur][spx] is None) or len(lst1)==stop_layer:
                    tot_sol1 += [lst1]
                    tot_sol2 += [lst2]
                else:
                    fffrom = ffrom[i][j][cur][spx]
                    if fffrom is None: return
                    mid = fffrom[0]
                    if mid=='single':
                        x = fffrom[1]
                        myprint(i,j,x,0,lst1,lst2)
                    elif fffrom[1]>=0:
                        x = fffrom[1]
                        y = fffrom[2]
                        myprint(i,mid,x,0,lst1,lst2)
                        myprint(mid+1,j,y,0,lst1,lst2)
                    else:
                        spx_n = -fffrom[1]
                        y = fffrom[2]
                        myprint(i,mid,cur,spx_n,lst1,lst2)
                        myprint(mid+1,j,y,0,lst1,lst2)
            if opt[i][j][p][q]>9999:
                return [None], None, None
            myprint(i,j,p,q,[],[])
            tot_sol = []
            for sol1, sol2 in zip(tot_sol1, tot_sol2):
                tot_sol += [ [val for pair in zip(sol2, sol1) for val in pair][::-1] ]
            assert tot_sol[-1][1] == -len(tot_sol)-1 and tot_sol[-1][-1]==-2, f"你是不是傻，最后一行的输出是{tot_sol[-1]}"

            return tot_sol, counts, opt_c[i][j][p][q]

        if double_seq is not None:
            assert False, "code reserved for other non-published use"

        if unique_path is not None:
            assert False, "code reserved for other non-published use"

        if no_debug and not full_sol:
            if unknown_root:
                sol = -1
                for i in range(self.count+1):
                    if opt[0][ll-1][i][0]==0:
                        if sol!=-1: print("multiple!")
                        sol = i
                return sol
            return opt[0][ll-1][self.all[0][0].id][0] #, optc[0][ll-1][self.all[0][0].id]

        if not no_debug:
            print(f"min_change = {opt[0][ll-1][self.all[0][0].id][0]}")
            #print(f"possibility = {optc[0][ll-1][self.all[0][0].id]}")

        tot_sol1, counts, num_poss = calc_dp_res(0,ll-1,self.all[0][0].id,0)
        if not no_debug:
            print(tot_sol1)

        best = 10000 if ll==0 else opt[0][ll-1][self.all[0][0].id][0]
        if full_sol:
            if cascade_count:
                return best, tot_sol1, counts, num_poss

            return best, tot_sol1

        return best 



    # DP solver for Non EQ conditional probability per token
    def solve_dp_prob_highprecision(self, aa, debug=True):
        from decimal import Decimal, getcontext
        getcontext().prec = 100
        getcontext().Emin = -999999
        getcontext().Emax = 999999
        if self.ptb is not None:
            return self.ptb.is_in_cfg_prob(aa)
        ll = len(aa)
        if True: # inside part
            opt = [ [ [ [0] for _ in range(self.count+1) ] for _ in range(ll)] for _ in range(ll)]
            look_left = [ [ [] for _ in range(self.count+1) ] for _ in range(ll)] 

            for k in range(ll):
                if aa[k] in self.idx[self.depth]:
                    opt[k][k][aa[k]][0] = 1
                    if (k-1) not in look_left[k][aa[k]]:
                        look_left[k][aa[k]] += [k-1]
            if debug: print('Performing DP1')
            for k in range(0, ll):
                if debug: print('.', end='', flush=True)
                for i in range(0,ll-k):
                    j = i+k
                    for a in reversed(range(self.depth)):
                        for b in range(self.sizes[a]):
                            cur = self.all[a][b].id
                            opt_pos = 0
                            pp = Decimal(1.0 / len(self.all[a][b].children))
                            for child in self.all[a][b].children:
                                if len(child)==1:
                                    x = child[0]
                                    x = self.all[a+1][x].id
                                    if opt[i][j][x][0]>0:
                                        if opt[i][j][cur][0]==0:
                                            look_left[j][cur] += [i-1]
                                        opt[i][j][cur][0] += pp * opt[i][j][x][0]
                                elif len(child)==2:
                                    x = child[0]
                                    y = child[1]
                                    x = self.all[a+1][x].id
                                    y = self.all[a+1][y].id
                                    #for mid in range(i, j):
                                    for mid in look_left[j][y]:
                                        if mid<i: break;
                                        if opt[i][mid][x][0] * opt[mid+1][j][y][0] > 0:
                                            if opt[i][j][cur][0]==0:
                                                look_left[j][cur] += [i-1]
                                            opt[i][j][cur][0] += pp * opt[i][mid][x][0] * opt[mid+1][j][y][0]
                                else:
                                    for i_c in range(len(child)-2):
                                        opt_pos+=1
                                        assert len(opt[i][j][cur])==opt_pos
                                        opt[i][j][cur] += [0]
                                        x = child[i_c]
                                        y = child[i_c+1]
                                        x = self.all[a+1][x].id
                                        y = self.all[a+1][y].id
                                        #for mid in range(i, j):
                                        for mid in look_left[j][y]:
                                            if mid<i: break;
                                            if i_c==0: 
                                                left_opt = opt[i][mid][x][0]
                                            else: 
                                                left_opt = opt[i][mid][cur][opt_pos-1]
                                                assert False, "CFG node length <= 3; haven't yet implemented"
                                            left_opt = Decimal(left_opt)
                                            if left_opt * opt[mid+1][j][y][0] > 0:
                                                opt[i][j][cur][opt_pos] += left_opt * opt[mid+1][j][y][0]
                                    x = child[-2]
                                    y = child[-1]
                                    x = self.all[a+1][x].id
                                    y = self.all[a+1][y].id
                                     #for mid in range(i, j):
                                    for mid in look_left[j][y]:
                                        if mid<i: break;
                                        if len(opt[i][mid][cur])<=opt_pos: continue
                                        left_opt = opt[i][mid][cur][opt_pos]
                                        if left_opt * opt[mid+1][j][y][0] > 0:
                                            if opt[i][j][cur][0]==0:
                                                look_left[j][cur] += [i-1]
                                            opt[i][j][cur][0] += pp * left_opt * opt[mid+1][j][y][0]
            if debug: print('')

            out = [ [ [0] for _ in range(self.count+1) ] for _ in range(ll+1)] # out[i][k] 代表 0...(i-1)之后加一个NT=k的可能概率
            out[0][self.all[0][0].id][0] = 1
            if debug: print('Performing DP2')
            for k in range(0, ll+1):
                if debug: print('x', end='', flush=True)
                # for i in range(0,ll-k):
                #     j = i+k
                if True:
                    i = None
                    j = None
                    for a in range(self.depth):
                        for b in range(self.sizes[a]):
                            cur = self.all[a][b].id
                            out_pos = 0
                            pp = Decimal(1.0 / len(self.all[a][b].children))
                            for child in self.all[a][b].children:
                                if len(child)==1:
                                    x = child[0]
                                    x = self.all[a+1][x].id
                                    out[k][x][0] += pp * out[k][cur][0]
                                elif len(child)==2:
                                    x = child[0]
                                    y = child[1]
                                    x = self.all[a+1][x].id
                                    y = self.all[a+1][y].id
                                    out[k][x][0] += pp * out[k][cur][0] # do only once
                                    for mid in range(k, ll):
                                    #for mid in look_left[j][y]:
                                        #if mid<i: break;
                                        # if out[i][mid][x][0] * out[mid+1][j][y][0] > 0:
                                        #     if out[i][j][cur][0]==0:
                                        #         look_left[j][cur] += [i-1]
                                        #     out[i][j][cur][0] += pp * out[i][mid][x][0] * out[mid+1][j][y][0]
                                        #out[i][mid][x][0] += pp * out[i][j][cur][0] * opt[mid+1][j][y][0]   # 往右走不需要opt
                                        #out[mid+1][j][y][0] += pp * out[i][j][cur][0] * opt[i][mid][x][0]
                                        out[mid+1][y][0] += pp * out[k][cur][0] * opt[k][mid][x][0]
                                else:
                                    for i_c in range(len(child)-2):
                                        out_pos+=1
                                        # assert len(out[i][j][cur])==out_pos
                                        # out[i][j][cur] += [0]
                                        x = child[i_c]
                                        y = child[i_c+1]
                                        x = self.all[a+1][x].id
                                        y = self.all[a+1][y].id
                                        if i_c==0:
                                            #out[k][x][0] += out[k][cur][out_pos] # do only once
                                            out[k][x][0] += pp * out[k][cur][0] # do only once
                                        for mid in range(k, ll):
                                        #for mid in look_left[j][y]:
                                            # if mid<i: break;
                                            # if i_c==0: 
                                            #     left_out = out[i][mid][x][0]
                                            # else: 
                                            #     left_out = out[i][mid][cur][out_pos-1]
                                            #     assert False, "CFG node length <= 3; haven't yet implemented"
                                            # if left_out * out[mid+1][j][y][0] > 0:
                                            #     out[i][j][cur][out_pos] += left_out * out[mid+1][j][y][0]
                                            
                                            if i_c==0:
                                                #out[mid+1][y][0] += out[k][cur][out_pos] * opt[k][mid][x][0]
                                                out[mid+1][y][0] += pp * out[k][cur][0] * opt[k][mid][x][0]
                                            else:
                                                assert False, "CFG node length <= 3; haven't yet implemented"
                                    x = child[-2]
                                    y = child[-1]
                                    x = self.all[a+1][x].id
                                    y = self.all[a+1][y].id
                                    #out[k][cur][out_pos] += pp * out[k][cur][0]
                                    for mid in range(k, ll):
                                    #for mid in look_left[j][y]:
                                        # if mid<i: break;
                                        # if len(out[i][mid][cur])<=out_pos: continue
                                        # left_out = out[i][mid][cur][out_pos]
                                        # if left_out * out[mid+1][j][y][0] > 0:
                                        #     if out[i][j][cur][0]==0:
                                        #         look_left[j][cur] += [i-1]
                                        #     out[i][j][cur][0] += pp * left_out * out[mid+1][j][y][0]
                                        out[mid+1][y][0] += pp * out[k][cur][0] * opt[k][mid][cur][out_pos]
            if debug: print('')

            terminals = [self.all[self.depth][b].id for b in range(self.sizes[self.depth])]
            terminals.sort()
            probs = []
            probs += [ [0.0] + [out[0][t][0] for t in terminals] ]
            probs_chosen = []
            Tinv = {}
            for i, t in enumerate(terminals):
                Tinv[t] = i+1
            for i in range(ll):
                cur = [ opt[0][i][self.all[0][0].id][0]  / out[i][aa[i]][0] ] # EOS probability
                probs_chosen += [probs[-1][Tinv[aa[i]]]]
                for t in terminals:
                    cur += [out[i+1][t][0] / out[i][aa[i]][0]]
                probs += [cur]
            probs_chosen += [probs[-1][0]]
            probs = [ [float(x) for x in probs[i]] for i in range(len(probs))]
            probs_chosen = [float(x) for x in probs_chosen]
            if debug:
                np.set_printoptions(suppress=True)
                print(f"Next is the groud-truth next-token prediction accuracy (if |T|=3 then each row has 4 numbers, first is EOS)")
                print(np.array(probs))
                print(f"Next should be all one (sanity check)")
                print(np.array(probs).sum(axis=1))
                print(f"Next is the probability for the given data to be generated according to CFG, may be extremely small")
                print(opt[0][ll-1][self.all[0][0].id][0])

            self.opt = opt
            self.out = out

        return probs, probs_chosen


