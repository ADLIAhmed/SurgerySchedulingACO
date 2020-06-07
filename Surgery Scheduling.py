#Importation des bibliothèques
import numpy as np
import pandas as pd
from numpy import inf
import time
import warnings
warnings.filterwarnings("ignore")


def Importation():
    #Importation des données de la table A3
    df=pd.read_excel("C:/Users/admin/Desktop/ROProject.xlsx")
    speciality=np.array(pd.DataFrame(df, columns=['speciality']))
    set_up=np.array(pd.DataFrame(df, columns=['set up time']))
    surgery=np.array(pd.DataFrame(df, columns=['surgery duration']))
    recovery=np.array(pd.DataFrame(df, columns=['recovery time']))

    #Création des matrices de temps et de spécialité
    time, sum=[], 0
    #Somme du temps du set_up, du recovery et de la chirurgie
    for i in range(0, len(surgery)):
        sum=set_up[i]+recovery[i]+surgery[i]
        time.append(sum)
    #Initialisation des matrices
    time_matrix=np.ones((44,44))
    time_matrix2=np.ones(44)
    speciality_matrix=np.ones(44)
    #Matrice de spécialités
    for i in range(0,44):
        speciality_matrix[i]=speciality[i]
    #Matrice de temps avec dimention 1
    for i in range(0,44):
        time_matrix2[i]=time[i]
    #Matrice de temps avec dimention 2 pour l'ACO
    for i in range(0,44):
        for j in range(0,44):
            if i==j:
                time_matrix[i][j]=0
            else:
                time_matrix[i][j]=time[i]
    d=time_matrix

def Surgery_Scheduling():
    #Initialition des variables de l'étude
    iteration, ants, surgeries, e,alpha, beta=100, 100, 44, .5, 1, 2
    m, n= ants, surgeries
    #Calcul de visibilité pour le prochain noeud
    surgeriesvis = 1/d
    surgeriesvis[surgeriesvis == inf ] = 0
    #Initialisation du pheromone dans le chemin de chaque chirurgie
    pheromne = .1*np.ones((m,n))
    #Initialisation du chemin des fourmis avec la taille du chemin
    #Le 1 a été ajouté puisqu'on veut revenir à la source de la chirurgie
    path = np.ones((m,n+1))
    for ite in range(iteration):
        #Initialisation de la position de début et de fin pour chaque fourmi
        path[:,0] = 1
        for i in range(m):
            #Création d'une copie de visibilité
            temp_surgeriesvis = np.array(surgeriesvis)
            for j in range(n-1):
                #Initialition du combine_feature
                combine_feature = np.zeros(44)
                #Initialition de la probabilité cummulative
                cum_prob = np.zeros(44)
                #Chirurgie actuelle pour chaque fourmi
                cur_loc = int(path[i,j]-1)
                #Placement la visibilité de la prochaine chirurgie en 0
                temp_surgeriesvis[:,cur_loc] = 0
                #Calcul du pheromone feature
                p_feature = np.power(pheromne[cur_loc,:],beta)
                #Calcul de visibilité feature
                v_feature = np.power(temp_surgeriesvis[cur_loc,:],alpha)
                #Ajout des axes pour s'adapter au problème
                p_feature = p_feature[:,np.newaxis]
                v_feature = v_feature[:,np.newaxis]
                #Calcul du combine feature
                combine_feature = np.multiply(p_feature,v_feature)
                #Somme de toutes les features
                total = np.sum(combine_feature)
                #Probabilité de chaque élément probs(i)
                probs = combine_feature/total
                #Calcul de la somme cummulatative
                cum_prob = np.cumsum(probs)
                #Nombre arbitraire entre 0 et 1
                r = np.random.random_sample()
                #Recherche la prochaine chirurgie ayant une probabilité supérieure à random(r)
                surgeryx = np.nonzero(cum_prob>r)[0][0]+1
                #Ajout de la chirurgie dans le chemin
                path[i,j+1] = surgeryx
            #Recherche de la dernière chirurgie dans le chemin non traversée
            left = list(set([i for i in range(1,n+1)])-set(path[i,:-2]))[0]
            #Ajout de la chirurgie non traversée dans le chemin
            path[i,-2] = left
        #Initialisation du chemin optimal
        path_opt = np.array(path)
        #Initialisation du temps total des tours en 0
        time_cost = np.zeros((m,1))
        for i in range(m):
            s = 0
            for j in range(n-1):
                #Calcul du temps total des tours
                s = s + d[int(path_opt[i,j])-1,int(path_opt[i,j+1])-1]
            #Stockage du temps du tour pour la ième fourmi dans la ième location
            time_cost[i]=s
        #Recherche de la location du temps minimal
        time_min_loc = np.argmin(time_cost)
        #Recherche du temps minimal
        time_min_cost = time_cost[time_min_loc]
        #Initialisation du présent chemin comme meilleur chemin
        best_scheduling = path[time_min_loc,:]
        #Evaboration du pheromone
        pheromne = (1-e)*pheromne
        for i in range(m):
            for j in range(n-1):
                #Mise à jour global des deltas et des To
                dt = 1/time_cost[i]
                pheromne[int(path_opt[i,j])-1,int(path_opt[i,j+1])-1] = pheromne[int(path_opt[i,j])-1,int(path_opt[i,j+1])-1] + dt
    best_scheduling=np.delete(best_scheduling,[0])

    #Initialisation des données de ressources
    Surgeons=[0,1,3,5,6,8,10,13,14,16,18,20]
    Nurses=20
    Anestheists=16
    Beds=10
    OR=[0,1,2,3,4,5,6,7,8,9,10]
    #Copie du meilleur chemin
    ressource_o=best_scheduling
    import time
    for i in ressource_o:
        #Initilialisation d'une liste Ressource avec des 0
        R=[0,0,0,0]
        #Récupération de la spacialité du chirurgien correspondant à la ième chirurgie
        x=int(speciality_matrix[int(i)-1])
        #Affectation des chirurgies
        if Surgeons[x]!=0:
            print("Surgery",int(i),"has a surgeon")
            Surgeons[x]=Surgeons[x]-1
            R[0]=1
        #Affectation des lits
        if Beds!=0:
            print("Surgery",int(i),"has a bed")
            Beds=Beds-1
            R[1]=1
        #Affectation des infirmiers
        if Nurses!=0:
            print("Surgery",int(i),"has 2 nurses")
            Nurses=Nurses-2
            R[2]=1
        #Affectation des anesthésistes
        if Anestheists!=0:
            print("Surgery",int(i),"has an Anestheist")
            Anestheists=Anestheists-1
            R[3]=1
        #Confirmation dans le cas de disponibilité de toutes les ressources
        if R==[1,1,1,1]:
            print("Surgery",int(i),"start now, ends after",time_matrix2[int(i)-1],"hour(s)")
            #Début de la chirurgie et calcul du temps
            start = time.clock()
            while time.clock() - start < time_matrix2[int(i)-1]:
                continue
            #Mise à jour des ressources une fois la chirurgie prend fin
            Surgeons[x]=Surgeons[x]+1
            Beds+=1
            Nurses=+2
            Anestheists=+1
        else:
            #Mise à jour des ressources
            Surgeons[x]=Surgeons[x]+1
            Beds+=1
            Nurses=+2
            Anestheists=+1

    #Affichage du meilleur scheduling
    print("\n\nThe optimal solution for this scheduling is :\n\n",best_scheduling)

    #Explication de l'affichage
    print("\nMeaning we're going to start in order with surgery number:\n" )
    for i in best_scheduling:
        if i==int(best_scheduling[0]):
            print(int(i)," then,",end='')
        elif i==int(best_scheduling[-2]):
            print(" ",int(i),"and finally, surgery number",end='')
        elif i==int(best_scheduling[-1]):
            print(" ",int(i),".")
        else:
            print(" ",int(i)," then,",end='')



