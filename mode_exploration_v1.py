

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import torch
from tqdm.notebook import tqdm


# ### Data Loading

# In[3]:


# # add angles
# df_clean["thetax_"] = np.arccos(df_clean.dircosx.values)*180/np.pi
# df_clean["thetaz_"]  = np.arccos(df_clean.dircosz.values)*180/np.pi
# df_clean["thetay_"] = np.arccos(df_clean.dircosy.values)*180/np.pi
# df_clean['v_'] = df_clean.apply(lambda x:x[['dircosx', 'dircosy', 'dircosz']].to_list(),axis=1)

# df_clean


# In[10]:


def _process_participant_tree(group):
    # take the group (per participant) and groupby startbpid
    # group must already have v_, theta_x,y,z in cols
    
    # get per branch point + delete trifurcations
    per_bp = group.groupby([pd.Grouper('idno'), pd.Grouper('startbpid')]).agg({
    lambda x:x.to_list() if len(x.to_list()) <=2 else np.nan}).reset_index()
    per_bp.columns = [a for a,b in per_bp.columns]

    # drop any rows with > 3 / flag (potentially need to remove this its an assumption!!!!)
    display(per_bp.loc[per_bp.endbpid.isnull()])
    per_bp.dropna(subset='endbpid', inplace=True)
    
    # now we have the tree grouped by branch point and need to get angles etc
    # add branch plane angle (returns vector normal to the plane containing the two child branches)
    def get_plane(x):
        if len(x) == 2:
            return np.cross(x[0], x[1])
        else:
#             print(len(x), x)
            return np.nan

    per_bp['plane_v'] = per_bp['v_'].apply(lambda x: get_plane(x))
    
    
    # find vector plane of the previous bifurcation
    def get_parent_vector_plane(x):
        v = np.array(per_bp['endbpid'].to_list()[1:]) # exclude trachea for the participant
        if np.any(np.isin(v,x.startbpid)):
            index = np.argmax(np.max(np.isin(v,x.startbpid), axis=1)) # find row that contains the parent branch (prev bpid)
#             print(index)
            try:
                row_parent = per_bp.iloc[index+1] # adding 1 back on for the trachea
            except:
                print("Help", index, len(per_bp))
            return row_parent.plane_v
        else:
            return np.nan
    # get rotation angle between previous split and current    
    def get_plane_rotation(x):
        dot_prod = np.dot(x.plane_v, x.parent_plane_v)/(np.linalg.norm(x.plane_v)*np.linalg.norm(x.parent_plane_v))
        rot_angle = np.arccos(dot_prod)*180/np.pi
        min_angle = np.min([rot_angle, 180-rot_angle])
        return min_angle
    per_bp['parent_plane_v'] = per_bp.apply(lambda x: get_parent_vector_plane(x), axis=1)
    per_bp['plane_rotation'] = per_bp.apply(
        lambda x: get_plane_rotation(x),
    axis=1)
    
    return per_bp



# In[11]:


# classify the bifurcations using branch angles, and rotations between successive branches
def _classify_mode(per_bp):
    '''
    Input: df grouped per branch point
    INITIAL MODES
    # mode orthog: angle between bp and previous bp is between 80-100 degrees and angle (min/max) ratio ~ 1 (bifurcation)
    # mode planar: angle between bp and previous bp is between -10 -10 degrees ~ 1 (bifurcation)
    # mode domain: angle ratio (min/max) < 0.75 and max angle < 160
    # mode other: not any of the above
    Returns: df grouped per branch point with classification of each branch point as [domain?, bifurcatinon?, other_mode?] and if bifurcation as [planar_bi, orthog_bi, other_bi]
    '''
    # classify based on pairs of angles
    per_bp['domain?'] = per_bp['angle'].apply(lambda x: True if ((np.max(x) > 160) & (np.min(x)/np.max(x)  <= 0.8)) else False) # one large angle
    per_bp['bifurcation?']= per_bp['angle'].apply(lambda x: True if np.min(x)/np.max(x) > 0.8 else False) # similar angles for children = bifurcation
    per_bp['other_mode?'] = ~(per_bp['domain?']|per_bp['bifurcation?'])
    per_bp['planar_bi'] = per_bp['bifurcation?'] & (per_bp['plane_rotation'] <= 20)
    per_bp['orthog_bi'] = per_bp['bifurcation?'] & (per_bp['plane_rotation'] >= 70)
    per_bp['other_bi'] = ~(per_bp['planar_bi']| per_bp['orthog_bi']) & (per_bp['bifurcation?'])
    
    return per_bp


# In[ ]:




def _iterate_participants(df_clean):
    # add angles
    df_clean["thetax_"] = np.arccos(df_clean.dircosx.values)*180/np.pi
    df_clean["thetaz_"]  = np.arccos(df_clean.dircosz.values)*180/np.pi
    df_clean["thetay_"] = np.arccos(df_clean.dircosy.values)*180/np.pi
    df_clean['v_'] = df_clean.apply(lambda x:x[['dircosx', 'dircosy', 'dircosz']].to_list(),axis=1)
    groups = df_clean.groupby('idno')
    results_list = []
    for name, group in tqdm(groups, desc='Iterating', display=True):
        per_bp_per_group = _process_participant_tree(group)
        per_bp = _classify_mode(per_bp_per_group)
        results_list.append(per_bp)
    total_df = pd.concat(results_list,axis=0)   
    total_df['gen'] = (total_df['weibel_generation'].apply(lambda x: np.nan if min(x) != max(x) else min(x)))
    print("Fixing generation", len(total_df))
    total_df.dropna(subset='gen', inplace=True)
    print(len(total_df))
    return total_df
# results = _iterate_participants(df)


# # In[ ]:


# total_df = pd.concat(results,axis=0)


# # In[ ]:


# total_df[['bifurcation?', 'domain?', 'other_mode?', 'planar_bi', 'orthog_bi', 'other_bi']].sum()/len(total_df)


# # In[ ]:


# total_df['gen'] = (total_df['weibel_generation'].apply(lambda x: np.nan if min(x) != max(x) else min(x)))
# total_df.dropna(subset='gen', inplace=True)
# print(len(total_df))


# # In[ ]:


# categories = total_df.groupby('gen').agg({
#     'domain?': sum,
#     'bifurcation?': sum,
#     'other_mode?':sum,
#     'planar_bi': sum,
#     'orthog_bi': sum,
#     'other_bi':sum,
#     'endbpid': "count"
# }).reset_index()
# categories.plot.bar(x='gen', y=['domain?', 'planar_bi','orthog_bi', 'other_bi', 'other_mode?'], stacked=True)


# # In[ ]:


# total_df.hist('gen')


# # In[ ]:


# w_lobes= total_df.explode(['lobes', 'angle', 'endbpid'])


# # In[ ]:


# w_lobes.groupby('lobes').agg(sum)[['domain?', 'other_mode?', 'planar_bi', 'orthog_bi', 'other_bi']].plot.bar(subplots=True, sharey=True,  ylabel='#segments', figsize=(10,10))


# # In[ ]:


# w_lobes


# # In[ ]:


# merged = df_clean.copy().merge(w_lobes[['idno', 'startbpid', 'endbpid', 'domain?', 'bifurcation?', 'other_mode?', 'planar_bi', 'orthog_bi', 'other_bi']],on=['idno', 'startbpid', 'endbpid'], how='left')


# # In[ ]:


# # merged[['idno', 'startbpid', 'endbpid', 'domain?', 'bifurcation?', 'other_mode?']].apply(lambda x:['domain?', 'bifurcation?', 'other_mode?'][np.argmax(x[['domain?', 'bifurcation?', 'other_mode?']])],axis=1)
# merged[['domain?', 'bifurcation?', 'other_mode?', 'planar_bi', 'orthog_bi', 'other_bi']].fillna(False, inplace=True)
# # merged[['domain?', 'bifurcation?', 'other_mode?']].dropna().astype(int).idxmax(axis=1)
# merged['category'] = merged[['domain?',  'other_mode?', 'planar_bi', 'orthog_bi', 'other_bi']].astype(int).idxmax(axis=1)


# # In[ ]:


# per_bp = df_clean.groupby([pd.Grouper('idno'), pd.Grouper('startbpid')]).agg({
#     lambda x:x.to_list() if len(x.to_list()) <=2 else np.nan
# }).reset_index()
# per_bp.columns = [a for a,b in per_bp.columns]

# # drop any rows with > 3 / flag
# display(per_bp.loc[per_bp.endbpid.isnull()])
# display(df_clean.loc[df_clean.startbpid == 83])
# per_bp.dropna(subset='endbpid', inplace=True)


# # In[ ]:


# per_bp


# # In[ ]:


# def get_parent_vector_plane(x):
#     v = np.array(per_bp['endbpid'].to_list()[1:]) # exclude trachea for the participant
#     if np.any(np.isin(v,x.startbpid)):
#         index = np.argmax(np.max(np.isin(v,x.startbpid), axis=1)) # find row that contains the parent branch (prev bpid)
#         print(index)
#         try:
#             row_parent = per_bp.iloc[index+1] # adding 1 back on for the trachea
#         except:
#             print("Help", index, len(per_bp))
#         return row_parent.plane_v
#     else:
#         return np.nan
# def get_plane_rotation(x):
#     dot_prod = np.dot(x.plane_v, x.parent_plane_v)/(np.linalg.norm(x.plane_v)*np.linalg.norm(x.parent_plane_v))
#     rot_angle = np.arccos(dot_prod)*180/np.pi
#     min_angle = np.min([rot_angle, 180-rot_angle])
#     return min_angle
# per_bp['parent_plane_v'] = per_bp.apply(lambda x: get_parent_vector_plane(x), axis=1)

# per_bp['plane_rotation'] = per_bp.apply(
#     lambda x: get_plane_rotation(x),
# axis=1)
# per_bp


# # In[ ]:


# # def dfs(graph, node, visited):
# #     if node not in visited:
# #         visited.append(node)
# #         for k in graph[node]:
# #             dfs(graph,k, visited)
# #     return visited

# # visited = dfs(graph1,'A', [])
# # print(visited)


# # In[ ]:


# # classify based on pairs of angles
# per_bp['domain?'] = per_bp['angle'].apply(lambda x: True if ((np.max(x) > 160) & (np.min(x)/np.max(x)  <= 0.8)) else False) # one large angle
# per_bp['bifurcation?']= per_bp['angle'].apply(lambda x: True if np.min(x)/np.max(x) > 0.8 else False) # similar angles for children = bifurcation
# per_bp['other_mode?'] = ~(per_bp['domain?']|per_bp['bifurcation?'])
# per_bp['planar_bi'] = per_bp['bifurcation?'] & (per_bp['plane_rotation'] <= 20)
# per_bp['orthog_bi'] = per_bp['bifurcation?'] & (per_bp['plane_rotation'] >= 70)
# per_bp['other_bi'] = ~(per_bp['planar_bi']| per_bp['orthog_bi']) & (per_bp['bifurcation?'])


# # In[ ]:





# # In[ ]:


# per_bp[['bifurcation?', 'domain?', 'other_mode?', 'planar_bi', 'orthog_bi', 'other_bi']].sum()/len(per_bp)


# # In[ ]:


# per_bp['gen'] = (per_bp['weibel_generation'].apply(lambda x: np.nan if min(x) != max(x) else min(x)))


# # In[ ]:


# categories = per_bp.groupby('gen').agg({
#     'domain?': sum,
#     'bifurcation?': sum,
#     'other_mode?':sum,
#     'planar_bi': sum,
#     'orthog_bi': sum,
#     'other_bi':sum,
#     'endbpid': "count"
# }).reset_index()
# categories.plot.bar(x='gen', y=['domain?', 'planar_bi','orthog_bi', 'other_bi', 'other_mode?'], stacked=True)


# # In[ ]:


# categories.plot.bar(x='gen', y=['domain?', 'planar_bi','orthog_bi', 'other_bi', 'other_mode?'], stacked=True)


# # In[ ]:


# df.groupby('weibel_generation').count()["idno"].plot.bar()


# # In[ ]:


# '''
# child's (dircosx, dircosy, dircosz) = relative position vector to add to parent vector to get child vector (unit vectors)
# so branching angle ==> np.dot(parent dircos, child dircos)*180/np.pi = cos(theta)
# angle is from heading dirn down (so its the major not minor angle) --> do 180-arccos(np.dot...) = branching angle

# iterate through branching points
# for each branching point in the list:
# - check first domain vs birfucation of some sort (angle diff)
# - for pairs of children do c1 cross c2 = vector pperp to plane containing the 2 vectors --> look at rotation between successive on path
# - so for pairs of children do first comparision of angles (Bi or Domain or OTHER) and calc plane for the pair
# - then iterate through the new df and look at adjacent pairs of and add marker for orthog / planar / other bifurcation

# '''
# parent =  df_clean.loc[df.endbpid==6]
# print(parent)
# parent_v = parent[['dircosx', 'dircosy', 'dircosz']].values.squeeze()
# print(parent_v.shape)
# child = df_clean.loc[df.endbpid==19]
# print(child)
# child_v = child[['dircosx', 'dircosy', 'dircosz']].values.squeeze()
# 180-np.arccos(np.dot(child_v, parent_v))*180/np.pi


if __name__ == "__main__":
    
    df_orig = pd.read_csv(os.path.abspath("/home/sneha/airway_exploration/e5_cleaned_v1.csv"))
    # few cleaning checks --> dropnas, fill remaining with 0s
    df = df.loc[~((df.dircosx.isnull()) & (df.startbpid != -1))]
    df.dropna(subset = ['idno', 'centerlinelength', 'startbpid', 'endbpid', 'weibel_generation'], inplace=True)
    df.fillna(0, inplace=True)
    print(f"Final clean df has {df.idno.nunique()} participants and a total length of {len(df)}")
    total_df = _iterate_participants(df)
    total_df.to_csv("per_branch_pt_mode_classification.csv", index=False)
    # In[ ]:


