# -*- coding: utf-8 -*-

"""
Created on Sun Jan  2 09:06:48 2022

@author: jcabrera

"""

import pandas as pd
import numpy as np
import pyomo.environ as pyo
import os
from plotly.offline import plot
import plotly.graph_objs as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
import math
pio.renderers.default = "browser"

# This function will return a
# list of positions where
# element exists in dataframe
def getIndexes(dfObj, value):
     
    # Empty list
    listOfPos = []
     
 
    # isin() method will return a dataframe with
    # boolean values, True at the positions   
    # where element exists
    result = dfObj.isin([value])
     
    # any() method will return
    # a boolean series
    seriesObj = result.any()
 
    # Get list of columns where element exists
    columnNames = list(seriesObj[seriesObj == True].index)
    
    # Iterate over the list of columns and
    # extract the row index where element exists
    for col in columnNames:
        rows = list(result[col][result[col] == True].index)
 
        for row in rows:
            listOfPos.append((row, col))
             
    # This list contains a list tuples with
    # the index of element in the dataframe
    return listOfPos

abs_path = os.getcwd()
file = abs_path + '\\' + 'DataHT_39_Congestion.xlsx'

df_Periodos     = pd.read_excel(file,sheet_name='Periodos')
df_Barra        = pd.read_excel(file,sheet_name='Barra')
df_DTerm        = pd.read_excel(file,sheet_name='DTerm')
df_DHidro       = pd.read_excel(file,sheet_name='DHidro')
df_DEmb         = pd.read_excel(file,sheet_name='DEmb')
df_CaudIn       = pd.read_excel(file,sheet_name='CaudIn')
df_PmaxT        = pd.read_excel(file,sheet_name='PmaxT')
df_PminT        = pd.read_excel(file,sheet_name='PminT')
df_Activa       = pd.read_excel(file,sheet_name='Activa')
df_PmaxH        = pd.read_excel(file,sheet_name='PmaxH')
df_PminH        = pd.read_excel(file,sheet_name='PminH')
df_Red          = pd.read_excel(file,sheet_name='Red')
df_CapLin       = pd.read_excel(file,sheet_name='CapLin')
df_MantLin      = pd.read_excel(file,sheet_name='MantLin')
df_Vmin         = pd.read_excel(file,sheet_name='Vmin')
df_Vmax         = pd.read_excel(file,sheet_name='Vmax')

for j in df_Red.index:
    df_Red.loc[j,'cos_phi'] = df_Red.loc[j,'R']/math.sqrt(df_Red.loc[j,'R']*df_Red.loc[j,'R'] + df_Red.loc[j,'X']*df_Red.loc[j,'X']) 

CostoRac = 6000 # $/MWh
Pbase = 100 # MVA

##%%
# Construcción de Ybus

nl = df_Red.loc[:,'Desde']
nr = df_Red.loc[:,'Hacia']

R = df_Red.loc[:,'R']
X = df_Red.loc[:,'X']
Bc = df_Red.loc[:,'1/2B']*1j
a = df_Red.loc[:,'tap'] 

nbr = df_Red.shape[0]
nbus = df_Barra.shape[0]

Z = R + X*1j
Y  = pd.Series(np.ones(nbr))/Z
df_Red['Y'] = -1*Y

Ybus = pd.DataFrame(0,index = df_Barra['Barra'].values.tolist(),columns = df_Barra['Barra'].values.tolist())

# Formación de elementos no diagonales

for k in df_Red.index:
    Ybus.loc[df_Red.loc[k,'Desde'],df_Red.loc[k,'Hacia']] = Ybus.loc[df_Red.loc[k,'Desde'],df_Red.loc[k,'Hacia']] - Y.loc[k]/a.loc[k]
    Ybus.loc[df_Red.loc[k,'Hacia'],df_Red.loc[k,'Desde']] = Ybus.loc[df_Red.loc[k,'Desde'],df_Red.loc[k,'Hacia']]

# Formación de elementos diagonales

for n in df_Barra.index:
    for k in df_Red.index:
        if df_Red.loc[k,'Desde'] == df_Barra.loc[n,'Barra']:
            Ybus.loc[df_Barra.loc[n,'Barra'],df_Barra.loc[n,'Barra']] = Ybus.loc[df_Barra.loc[n,'Barra'],df_Barra.loc[n,'Barra']] + Y.loc[k]/(a.loc[k]*a.loc[k]) + Bc.loc[k]
        elif df_Red.loc[k,'Hacia'] == df_Barra.loc[n,'Barra']:
            Ybus.loc[df_Barra.loc[n,'Barra'],df_Barra.loc[n,'Barra']] = Ybus.loc[df_Barra.loc[n,'Barra'],df_Barra.loc[n,'Barra']] + Y.loc[k] + Bc.loc[k]

##############################################################################
##%%

model = pyo.ConcreteModel()
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
model.name = 'HT_Red_CP'

#Declaración de sets

model.st     = pyo.Set(initialize = df_Periodos['Periodos'].values.tolist(),ordered = True)
model.sBarra = pyo.Set(initialize = df_Barra['Barra'].values.tolist())
model.sTerm  = pyo.Set(initialize = df_DTerm['CT'].values.tolist())
model.sHidro = pyo.Set(initialize = df_DHidro['CH'].values.tolist())
model.sEmb   = pyo.Set(initialize = df_DEmb['Embalse'].values.tolist())
model.sCaud  = pyo.Set(initialize = df_CaudIn['Caudal'].values.tolist())
model.sLin   = pyo.Set(initialize = df_Red['Linea'].values.tolist())

# Declaración de sets de mapeo

model.sMaptb = pyo.Set(initialize = df_DTerm[['CT','Barra']].values.tolist())
model.sMaphb = pyo.Set(initialize = df_DHidro[['CH','Barra']].values.tolist())
model.sMaplbb = pyo.Set(initialize = df_Red[['Linea','Desde','Hacia']].values.tolist())
model.sMaphemb = pyo.Set(initialize = df_DHidro[['CH','Embalse']].values.tolist())
model.sMapembq = pyo.Set(initialize = df_DEmb[['Embalse','Caudal Ingreso']].values.tolist())

# model.st.display()
# model.sBarra.display()
# model.sTerm.display()
# model.sHidro.display()
# model.sEmb.display()
# model.sCaud.display()
# model.sLin.display()
# model.sMaptb.display() 
# model.sMaphb.display() 
# model.sMaplbb.display()
# model.sMaphemb.display()
# model.sMapembq.display()

# Declaración de variables

model.vPt    = pyo.Var(model.sTerm , model.st, domain=pyo.NonNegativeReals,initialize=0 )
model.vPt.setlb(0)
for Term in model.sTerm:
    for t in model.st:
        model.vPt[Term,t].setub(df_PmaxT.loc[getIndexes(df_PmaxT, Term)[0][0],t])
#model.vPt.display()

model.vPh    = pyo.Var(model.sHidro , model.st, domain=pyo.NonNegativeReals,initialize=0 )
model.vPh.setlb(0)
for Hidro in model.sHidro:
    for t in model.st:
        model.vPh[Hidro,t].setub(df_PmaxH.loc[getIndexes(df_PmaxH, Hidro)[0][0],t])
#model.vPh.display()

model.vV     = pyo.Var(model.sEmb , model.st,initialize=0)
model.vV.setlb(0)
for Emb in model.sEmb:
    for t in model.st:
        model.vV[Emb,t].setub(df_Vmax.loc[getIndexes(df_Vmax, Emb)[0][0],t])
        model.vV[Emb,t].setlb(df_Vmin.loc[getIndexes(df_Vmin, Emb)[0][0],t])
#model.vV.display()

model.vSp    = pyo.Var(model.sEmb , model.st, domain=pyo.NonNegativeReals,initialize=0)
model.vSp.setlb(0)
model.vSp.setub(100)
#model.vSp.display()

model.vd     = pyo.Var(model.sBarra , model.st,initialize=0)
model.vd.setlb(-60)
model.vd.setub(60)
# for t in model.st:
#     model.vd[model.sBarra.first(),t].fix(0) 
#model.vd.display()

model.vpr    = pyo.Var(model.sBarra , model.st, domain=pyo.NonNegativeReals ,initialize=0)
model.vpr.setlb(0)
for Barra in model.sBarra:
    for t in model.st:    
        #print(Barra + ' - ' + t + ' - ' + str(df_Activa.loc[getIndexes(df_Activa, Barra)[0][0],t]))
        model.vpr[Barra,t].setub(df_Activa.loc[getIndexes(df_Activa, Barra)[0][0],t])
#model.vpr.display()


model.vEO = pyo.Var(model.sTerm , model.st, domain=pyo.Binary,initialize=0)
#model.vEOpr = pyo.Var(model.sBarra , model.st, domain=pyo.Binary)
model.vEArr = pyo.Var(model.sTerm , model.st, domain=pyo.Binary,initialize=0)


#model.vEO.display()
#model.vEArr.display()

# Función objetivo

model.oCosto = pyo.Objective(expr = (sum(sum(  (df_DTerm.loc[getIndexes(df_DTerm, Term)[0][0],'A']*model.vEO[Term,t] 
                                            + df_DTerm.loc[getIndexes(df_DTerm, Term)[0][0],'B']*model.vPt[Term,t] ) for Term in model.sTerm ) for t in model.st ) 
                                    + sum(sum(model.vPh[Hidro,t]*df_DHidro.loc[getIndexes(df_DHidro, Hidro)[0][0],'COyM'] for Hidro in model.sHidro) for t in model.st )
                                    + sum(sum(model.vpr[Barra,t]*CostoRac for Barra in model.sBarra) for  t in model.st) 
                                    + sum(sum(model.vEArr[Term,t]*df_DTerm.loc[getIndexes(df_DTerm, Term)[0][0],'CArr'] for Term in model.sTerm) for t in model.st)
                                    )/1000
                            
                             )
#+ df_DTerm.loc[getIndexes(df_DTerm, Term)[0][0],'C']*model.vPt[Term,t]*model.vPt[Term,t]
#model.oCosto.display()

# Limites de variables

# Máximo corte en barra
# model.eprmax= pyo.ConstraintList()
# for Barra in model.sBarra:
#     for t in model.st:        
#         model.eprmax.add( model.vpr[Barra,t] - df_Activa.loc[getIndexes(df_Activa, Barra)[0][0],t] <= 0) #*model.vEOpr[Barra,t]
#model.eprmax.display()

# Potencia térmica mínima 
model.ePtmin = pyo.ConstraintList()
for Term in model.sTerm:
    for t in model.st:        
        model.ePtmin.add( df_PminT.loc[getIndexes(df_PminT, Term)[0][0],t]*model.vEO[Term,t] - model.vPt[Term,t] <= 0)
#model.ePtmin.display()

# Potencia térmica máxima 
model.ePtmax = pyo.ConstraintList()
for Term in model.sTerm:    
    for t in model.st:        
        model.ePtmax.add(model.vPt[Term,t] -  df_PmaxT.loc[getIndexes(df_PmaxT, Term)[0][0],t]*model.vEO[Term,t] <= 0)
#model.ePtmax.display()

# Potencia hidro mínima 
model.ePhmin = pyo.ConstraintList()
for Hidro in model.sHidro:    
    for t in model.st:        
        model.ePhmin.add(df_PminH.loc[getIndexes(df_PminH, Hidro)[0][0],t] - model.vPh[Hidro,t] <= 0)
#model.ePhmin.display()

# Potencia hidro máxima 
model.ePhmax = pyo.ConstraintList()
for Hidro in model.sHidro:
    for t in model.st:
        model.ePhmax.add(model.vPh[Hidro,t] -  df_PmaxH.loc[getIndexes(df_PmaxH, Hidro)[0][0],t] <= 0)
#model.ePhmax.display()

# # Volumen minimo
# model.eVmin = pyo.ConstraintList()
# for Emb in model.sEmb:    
#     for t in model.st:        
#         model.eVmin.add(  df_Vmin.loc[getIndexes(df_Vmin, Emb)[0][0],t] - model.vV[Emb,t] <= 0)
# #model.eVmin.display()

# # Volumen máximo 
# model.eVmax = pyo.ConstraintList()
# for Emb in model.sEmb:
#     for t in model.st:
#         model.eVmax.add(model.vV[Emb,t] - df_Vmax.loc[getIndexes(df_Vmax, Emb)[0][0],t] <= 0)
# #model.eVmax.display()

# =============================================================================
# SECCIÓN DE ECUACIONES
# =============================================================================

# Ecuación de balance nodal
model.eBalNod = pyo.ConstraintList()
for Barra in model.sBarra:                                
    for t in model.st:
        model.eBalNod.add(  sum( model.vPt[Term,t] for Term in model.sTerm if (Term,Barra) in model.sMaptb) 
                          + sum( model.vPh[Hidro,t] for Hidro in model.sHidro if (Hidro,Barra) in model.sMaphb)
                          + model.vpr[Barra,t] == df_Activa.loc[getIndexes(df_Activa, Barra)[0][0],t]                           
                          + sum( Pbase*(((model.vd[Barra,t]*math.pi/180 - model.vd[Barra_par,t]*math.pi/180)*(model.vd[Barra,t]*math.pi/180 - model.vd[Barra_par,t]*math.pi/180))/2)*(Ybus.loc[Barra,Barra_par].imag)*(abs((1/Ybus.loc[Barra,Barra_par]).real)/(abs(1/Ybus.loc[Barra,Barra_par]))) for Barra_par in model.sBarra if (Barra_par != Barra) & (abs(Ybus.loc[Barra,Barra_par]) != 0))
                          + sum( Pbase*(model.vd[Barra,t]*math.pi/180 - model.vd[Barra_par,t]*math.pi/180)*(Ybus.loc[Barra,Barra_par].imag) for Barra_par in model.sBarra if (Barra_par != Barra) & (abs(Ybus.loc[Barra,Barra_par]) != 0))                          
                          )





#+ sum( Pbase*(((model.vd[Barra,t]*math.pi/180 - model.vd[Barra_par,t]*math.pi/180)*(model.vd[Barra,t]*math.pi/180 - model.vd[Barra_par,t]*math.pi/180))/2)*(Ybus.loc[Barra,Barra_par].imag)*(abs((1/Ybus.loc[Barra,Barra_par]).real)/(abs(1/Ybus.loc[Barra,Barra_par]))) for Barra_par in model.sBarra if (Barra_par != Barra) & (abs(Ybus.loc[Barra,Barra_par]) != 0))


# + sum( Pbase*(((model.vd[Barra,t]*math.pi/180 - model.vd[Barra_par,t]*math.pi/180)*(model.vd[Barra,t]*math.pi/180 - model.vd[Barra_par,t]*math.pi/180))/2)*(Ybus.loc[Barra,Barra_par].imag)*(abs((1/Ybus.loc[Barra,Barra_par]).real)/(abs(1/Ybus.loc[Barra,Barra_par]))) for Barra_par in model.sBarra if (Barra_par != Barra) & (abs(Ybus.loc[Barra,Barra_par]) != 0))
#                          + sum( Pbase*(((model.vd[Barra,t]*math.pi/180 - model.vd[Barra_par,t]*math.pi/180)*(model.vd[Barra,t]*math.pi/180 - model.vd[Barra_par,t]*math.pi/180))/2)*(Ybus.loc[Barra,Barra_par].imag)*(abs((1/Ybus.loc[Barra,Barra_par]).real)/(abs(1/Ybus.loc[Barra,Barra_par]))) for Barra_par in model.sBarra if (Barra_par != Barra) & (abs(Ybus.loc[Barra,Barra_par]) != 0))

#+ sum( Pbase*(1-pyo.cos(model.vd[Barra,t]*math.pi/180 - model.vd[Barra_par,t]*math.pi/180))*(Ybus.loc[Barra,Barra_par].imag)*(abs((1/Ybus.loc[Barra,Barra_par]).real)/(abs(1/Ybus.loc[Barra,Barra_par]))) for Barra_par in model.sBarra if (Barra_par != Barra) & (abs(Ybus.loc[Barra,Barra_par]) != 0))

# model.eBalNod.display()        

#Ecuación de continuidad del embalse
model.eBalEmb = pyo.ConstraintList()
for Emb in model.sEmb:
    for t in model.st:
        if t == model.st.first():
            model.eBalEmb.add( model.vV[Emb,t] == df_DEmb.loc[getIndexes(df_DEmb, Emb)[0][0],'Vini'] 
                              + 3.6*( sum( df_CaudIn.loc[getIndexes(df_CaudIn, Caud)[0][0],t] for Caud in model.sCaud if (Emb,Caud) in model.sMapembq) 
                                     - sum( model.vPh[Hidro,t]/(df_DHidro.loc[getIndexes(df_DHidro, Hidro)[0][0],'CoefProd']) for Hidro in model.sHidro if (Hidro,Emb) in model.sMaphemb )
                                     - model.vSp[Emb,t]
                                     )
                              )
        elif t != model.st.first():
            model.eBalEmb.add( model.vV[Emb,t] == model.vV[Emb,model.st.prev(t)] 
                              + 3.6*( sum( df_CaudIn.loc[getIndexes(df_CaudIn, Caud)[0][0],t] for Caud in model.sCaud if (Emb,Caud) in model.sMapembq) 
                                     - sum( model.vPh[Hidro,t]/(df_DHidro.loc[getIndexes(df_DHidro, Hidro)[0][0],'CoefProd']) for Hidro in model.sHidro if (Hidro,Emb) in model.sMaphemb )
                                     - model.vSp[Emb,t]
                                     )
                              )            
            
#Ecuación de vmeta del embalse
model.eVmetaEmb = pyo.ConstraintList()
for Emb in model.sEmb:
    for t in model.st:            
        if t == model.st.last():
            if df_DEmb.loc[getIndexes(df_DEmb, Emb)[0][0],'Vmeta_considera'] == 1:
                model.eVmetaEmb.add( df_DEmb.loc[getIndexes(df_DEmb, Emb)[0][0],'Vmeta'] == model.vV[Emb,t])     

#Ecuacion de capacidad de lineas de transmisión

model.eCapLinPos = pyo.ConstraintList()
for Lin in model.sLin:
    for t in model.st:
        temp_idx = getIndexes(df_Red, Lin)[0][0]
        temp_idx_cap = getIndexes(df_CapLin, Lin)[0][0]
        model.eCapLinPos.add( Pbase*(model.vd[df_Red.loc[temp_idx,'Desde'],t]*math.pi/180 - model.vd[df_Red.loc[temp_idx,'Hacia'],t]*math.pi/180)*(df_Red.loc[temp_idx,'Y'].imag) - df_CapLin.loc[temp_idx_cap,t] <= 0)

model.eCapLinNeg = pyo.ConstraintList()
for Lin in model.sLin:
    for t in model.st:
        temp_idx = getIndexes(df_Red, Lin)[0][0]
        temp_idx_cap = getIndexes(df_CapLin, Lin)[0][0]
        model.eCapLinNeg.add(  -1*df_CapLin.loc[temp_idx_cap,t] - Pbase*(model.vd[df_Red.loc[temp_idx,'Desde'],t]*math.pi/180-model.vd[df_Red.loc[temp_idx,'Hacia'],t]*math.pi/180)*(df_Red.loc[temp_idx,'Y'].imag)  <= 0 )

# Ecuacion de condición térmica inicial
for Term in model.sTerm:
    temp_idx = getIndexes(df_DTerm, Term)[0][0]
    CondInic_temp = df_DTerm.loc[temp_idx,'CondInic']
    if CondInic_temp == 1: #Conectada
        if df_DTerm.loc[temp_idx,'T_inic'] < df_DTerm.loc[temp_idx,'TminOp']:
            Rest_h = df_DTerm.loc[temp_idx,'TminOp'] - df_DTerm.loc[temp_idx,'T_inic']
            for u in range(1,Rest_h+1):
                #print(u)
                model.vEO[Term,model.st.at(u)].fix(1)
    if CondInic_temp == 0: # Desconectada
        if df_DTerm.loc[temp_idx,'T_inic'] < df_DTerm.loc[temp_idx,'TarrSus']:
            Rest_h = df_DTerm.loc[temp_idx,'TarrSus'] - df_DTerm.loc[temp_idx,'T_inic']
            for u in range(1,Rest_h+1):
                #print(u)
                model.vEO[Term,model.st.at(u)].fix(0)

# Ecuacion del tiempo minimo de operacion
model.eTminOp = pyo.ConstraintList()
for Term in model.sTerm:
    temp_idx = getIndexes(df_DTerm, Term)[0][0]
    TminOP = df_DTerm.loc[temp_idx,'TminOp']
    for t in model.st:
        if (t != model.st.first()) & (t != model.st.last()):
            for k in model.st:
                if (model.st.ord(t) + TminOP-1) >= model.st.ord(model.st.last()):
                    max_st = model.st.last()
                else:
                    max_st = model.st.at(model.st.ord(t) + TminOP-1) 
                if (model.st.ord(k) >= model.st.ord(model.st.next(t))) & ( model.st.ord(k) <= model.st.ord(max_st) ):
                    model.eTminOp.add( model.vEO[Term,model.st.prev(t)]-model.vEO[Term,t] + model.vEO[Term,k] >= 0 )

# Ecuacion del tiempo entre arranques suscesivos
model.eTarrSus = pyo.ConstraintList()
for Term in model.sTerm:
    temp_idx = getIndexes(df_DTerm, Term)[0][0]
    TarrSus = df_DTerm.loc[temp_idx,'TarrSus']
    for t in model.st:
        if (t != model.st.first()) & (t != model.st.last()):
            for k in model.st:
                if (model.st.ord(t) + TarrSus-1) >= model.st.ord(model.st.last()):
                    max_st = model.st.last()
                else:
                    max_st = model.st.at(model.st.ord(t) + TarrSus-1) 
                if (model.st.ord(k) >= model.st.ord(model.st.next(t))) & ( model.st.ord(k) <= model.st.ord(max_st) ):
                    model.eTarrSus.add( model.vEO[Term,model.st.prev(t)]-model.vEO[Term,t] + model.vEO[Term,k] <= 1 )

# Ecuacion de costo de arranque
model.eCArr = pyo.ConstraintList()
for Term in model.sTerm:
    temp_idx = getIndexes(df_DTerm, Term)[0][0]
    EOinic = df_DTerm.loc[temp_idx,'CondInic']
    for t in model.st:
        if model.st.ord(t) == 1:
            model.eCArr.add(model.vEArr[Term,t] + model.vEO[Term,t] + EOinic <= 2)
            model.eCArr.add(model.vEArr[Term,t] - model.vEO[Term,t] + EOinic >= 0)
        else:
            model.eCArr.add(model.vEArr[Term,t] + model.vEO[Term,t] + model.vEO[Term,model.st.prev(t)] <= 2)
            model.eCArr.add(model.vEArr[Term,t] - model.vEO[Term,t] + model.vEO[Term,model.st.prev(t)] >= 0)
                
# =============================================================================
# SECCION PARA SOLUCION DEL PROBLEMA
# =============================================================================
# Solucion del problema 

# opt_bonmin = pyo.SolverFactory('bonmin') 
# results = opt_bonmin.solve(model,tee=True)
# model.vEO.fix()
# model.vEArr.fix()
# opt = pyo.SolverFactory('ipopt') 
# results = opt.solve(model,tee=True) #, tee=True
# model.vEO.free()
# model.vEArr.free()

opt = pyo.SolverFactory('couenne') 
results = opt.solve(model,tee=True) #, tee=True
result_couenne = results
status = results.solver.termination_condition.value

if status != 'optimal':
    bool_status = 0
elif status == 'optimal':
    bool_status = 1
    model.vEO.fix()
    model.vEArr.fix()
    
    opt = pyo.SolverFactory('ipopt') 
    results = opt.solve(model,tee=True) #, tee=True
    status = results.solver.termination_condition.value        

if bool_status == 1:
    
    # for c in model.component_objects(pyo.Constraint, active = True):
    #     print(c)
    
    # rdf_eprmax = pd.DataFrame(np.nan,index = model.st, columns = model.sBarra)    
    # key_cont = 1   
    # for Barra in model.sBarra:
    #     for t in model.st:  
    #         rdf_eprmax.loc[t,Barra] = model.dual[model.eprmax[key_cont]]*1000
    #         key_cont = key_cont + 1
    # del key_cont

    rdf_ePtmin = pd.DataFrame(np.nan,index = model.st, columns = model.sTerm)    
    key_cont = 1   
    for Term in model.sTerm:
        for t in model.st: 
            rdf_ePtmin.loc[t,Term] = model.dual[model.ePtmin[key_cont]]*1000
            key_cont = key_cont + 1
    del key_cont    
    
    rdf_ePtmax = pd.DataFrame(np.nan,index = model.st, columns = model.sTerm)    
    key_cont = 1   
    for Term in model.sTerm:
        for t in model.st: 
            rdf_ePtmax.loc[t,Term] = model.dual[model.ePtmax[key_cont]]*1000
            key_cont = key_cont + 1
    del key_cont        

    rdf_ePhmin = pd.DataFrame(np.nan,index = model.st, columns = model.sHidro)    
    key_cont = 1   
    for Hidro in model.sHidro:
        for t in model.st: 
            rdf_ePhmin.loc[t,Hidro] = model.dual[model.ePhmin[key_cont]]*1000
            key_cont = key_cont + 1
    del key_cont  

    rdf_ePhmax = pd.DataFrame(np.nan,index = model.st, columns = model.sHidro)    
    key_cont = 1   
    for Hidro in model.sHidro:
        for t in model.st: 
            rdf_ePhmax.loc[t,Hidro] = model.dual[model.ePhmax[key_cont]]*1000
            key_cont = key_cont + 1
    del key_cont  

    # rdf_eVmin = pd.DataFrame(np.nan,index = model.st, columns = model.sEmb)     
    # key_cont = 1     
    # for Emb in model.sEmb:    
    #     for t in model.st:  
    #         rdf_eVmin.loc[t,Emb] = model.dual[model.eVmin[key_cont]]*1000
    #         key_cont = key_cont + 1
    # del key_cont  

    # rdf_eVmax = pd.DataFrame(np.nan,index = model.st, columns = model.sEmb)     
    # key_cont = 1 
    # for Emb in model.sEmb:    
    #     for t in model.st:  
    #         rdf_eVmax.loc[t,Emb] = model.dual[model.eVmax[key_cont]]*1000
    #         key_cont = key_cont + 1
    # del key_cont  
    
    rdf_eBalNod = pd.DataFrame(np.nan,index = model.st, columns = model.sBarra)
    key_cont = 1
    for Barra in model.sBarra:                                
        for t in model.st:
            rdf_eBalNod.loc[t,Barra] = model.dual[model.eBalNod[key_cont]]*1000
            key_cont = key_cont + 1
    del key_cont


    rdf_eBalEmb = pd.DataFrame(np.nan,index = model.st, columns = model.sEmb)
    key_cont = 1    
    for Emb in model.sEmb:
        for t in model.st:
            rdf_eBalEmb.loc[t,Emb] = model.dual[model.eBalEmb[key_cont]]*1000
            key_cont = key_cont + 1
    del key_cont
            
    rdf_eVmetaEmb = pd.DataFrame(np.nan,index = model.st, columns = model.sEmb)
    key_cont = 1    
    for Emb in model.sEmb:
        for t in model.st:
            if t == model.st.last():
                if df_DEmb.loc[getIndexes(df_DEmb, Emb)[0][0],'Vmeta_considera'] == 1:            
                    rdf_eVmetaEmb.loc[t,Emb] = model.dual[model.eVmetaEmb[key_cont]]*1000
                    key_cont = key_cont + 1
    del key_cont    

    rdf_eCapLinPos = pd.DataFrame(np.nan,index = model.st, columns = model.sLin)
    key_cont = 1    
    for Lin in model.sLin:
        for t in model.st:
            rdf_eCapLinPos.loc[t,Lin] = model.dual[model.eCapLinPos[key_cont]]*1000
            key_cont = key_cont + 1
    del key_cont    

    rdf_eCapLinNeg = pd.DataFrame(np.nan,index = model.st, columns = model.sLin)
    key_cont = 1    
    for Lin in model.sLin:
        for t in model.st:
            rdf_eCapLinNeg.loc[t,Lin] = model.dual[model.eCapLinNeg[key_cont]]*1000
            key_cont = key_cont + 1
    del key_cont    


    print('-----------------SE ENCONTRÓ LA SOLUCIÓN OPTIMA-----------------')

# rdf_eBalNod = pd.DataFrame(np.nan,index = model.st, columns = model.sBarra)
# for t in model.st:
#     for Barra in model.sBarra:
#         rdf_d.loc[t,Barra] = model.vd[Barra,t]*math.pi/180.value

# =============================================================================
# SECCION PARA IMPRESIÓN DE LOS RESULTADOS
# =============================================================================

status = results.solver.termination_condition.value

rdf_Costo_Operativo = model.oCosto.expr()   

rdf_Pt = pd.DataFrame(np.nan,index = model.st, columns = model.sTerm )
for t in model.st:
    for Term in model.sTerm:
        rdf_Pt.loc[t,Term] = model.vPt[Term,t].value

rdf_Ph = pd.DataFrame(np.nan,index = model.st, columns = model.sHidro )
for t in model.st:
    for Hidro in model.sHidro:
        rdf_Ph.loc[t,Hidro] = model.vPh[Hidro,t].value

rdf_d = pd.DataFrame(np.nan,index = model.st, columns = model.sBarra )
for t in model.st:
    for Barra in model.sBarra:
        rdf_d.loc[t,Barra] = model.vd[Barra,t].value
        
#rdf_d_grados = rdf_d*180/3.1415926        

rdf_pr = pd.DataFrame(np.nan,index = model.st, columns = model.sBarra )
for t in model.st:
    for Barra in model.sBarra:
        rdf_pr.loc[t,Barra] = model.vpr[Barra,t].value

rdf_flujo = pd.DataFrame(np.nan,index = model.st, columns = model.sLin )
for t in model.st:
    for Lin in model.sLin:
        temp_idx = getIndexes(df_Red, Lin)[0][0]
        rdf_flujo.loc[t,Lin] = Pbase*(model.vd[df_Red.loc[temp_idx,'Desde'],t].value*math.pi/180-model.vd[df_Red.loc[temp_idx,'Hacia'],t].value*math.pi/180)*(df_Red.loc[temp_idx,'Y'].imag)

rdf_perdidas = pd.DataFrame(np.nan,index = model.st, columns = model.sLin )
for t in model.st:
    for Lin in model.sLin:
        temp_idx = getIndexes(df_Red, Lin)[0][0]
        rdf_perdidas.loc[t,Lin] = Pbase*2*(1-math.cos(model.vd[df_Red.loc[temp_idx,'Desde'],t].value*math.pi/180-model.vd[df_Red.loc[temp_idx,'Hacia'],t].value*math.pi/180))*(abs((1/df_Red.loc[temp_idx,'Y']).real)/(abs(1/df_Red.loc[temp_idx,'Y'])))*(df_Red.loc[temp_idx,'Y'].imag)
                                                                                    
rdf_V = pd.DataFrame(np.nan,index = model.st, columns = model.sEmb )
for t in model.st:
    for Emb in model.sEmb:
        rdf_V.loc[t,Emb] = model.vV[Emb,t].value

rdf_Sp = pd.DataFrame(np.nan,index = model.st, columns = model.sEmb )
for t in model.st:
    for Emb in model.sEmb:
        rdf_Sp.loc[t,Emb] = model.vSp[Emb,t].value

rdf_EO = pd.DataFrame(np.nan,index = model.st, columns = model.sTerm )
for t in model.st:
    for Term in model.sTerm:
        rdf_EO.loc[t,Term] = model.vEO[Term,t].value

# rdf_EOpr = pd.DataFrame(np.nan,index = model.st, columns = model.sBarra )
# for t in model.st:
#     for Barra in model.sBarra:
#         rdf_EOpr.loc[t,Barra] = model.vEOpr[Barra,t].value

rdf_EArr = pd.DataFrame(np.nan,index = model.st, columns = model.sTerm )
for t in model.st:
    for Term in model.sTerm:
        rdf_EArr.loc[t,Term] = model.vEArr[Term,t].value


# =============================================================================
# SECCION PARA GRAFICOS
# =============================================================================
Texto = "Resultados_de_Simulacion_Congestion_1.0"
fig = make_subplots(rows=2, 
                    cols=3,
                    subplot_titles = ['Flujo en lineas',
                                      'Pérdidas',
                                      'Costo Marginal',
                                      'Potencia Activa',
                                      'Racionamiento',
                                      'Volumen de embalses',
                                      ])

fig.layout.title.text = Texto

for j in rdf_flujo.columns:    
    fig.add_trace( go.Scatter(x = rdf_flujo.index, 
                              y = rdf_flujo[j], 
                              mode="lines", 
                              showlegend=True,
                              name = str(j)) , row=1, col=1)    
fig.update_xaxes(title_text="Bloques horarios", row = 1, col = 1)
fig.update_yaxes(title_text="Flujo en lineas [MW]", row = 1, col = 1)

for j in rdf_perdidas.columns:    
    fig.add_trace( go.Scatter(x = rdf_perdidas.index, 
                              y = rdf_perdidas[j], 
                              mode="lines", 
                              showlegend=True,
                              name = str(j)) , row=1, col=2)    
fig.update_xaxes(title_text="Bloques horarios", row = 1, col = 2)
fig.update_yaxes(title_text="Pérdidas en lineas [MW]", row = 1, col = 2)

for j in rdf_eBalNod.columns:    
    fig.add_trace( go.Scatter(x = rdf_eBalNod.index, 
                              y = rdf_eBalNod[j], 
                              mode="lines", 
                              showlegend=True,
                              name = str(j)) , row=1, col=3)    
fig.update_xaxes(title_text="Bloques horarios", row = 1, col = 3)
fig.update_yaxes(title_text="Costos marginales [$/MWh]", row = 1, col = 3)


for j in rdf_Ph.columns:    
    fig.add_trace( go.Scatter(x = rdf_Ph.index, 
                              y = rdf_Ph[j], 
                              mode="lines", 
                              showlegend=True,
                              name = str(j)) , row=2, col=1)    
for j in rdf_Pt.columns:    
    fig.add_trace( go.Scatter(x = rdf_Pt.index, 
                              y = rdf_Pt[j], 
                              mode="lines", 
                              showlegend=True,
                              name = str(j)) , row=2, col=1)        
fig.update_xaxes(title_text="Bloques horarios", row = 2, col = 1)
fig.update_yaxes(title_text="Costos marginales [$/MWh]", row = 2, col = 1)

for j in rdf_pr.columns:    
    fig.add_trace( go.Scatter(x = rdf_pr.index, 
                              y = rdf_pr[j], 
                              mode="lines", 
                              showlegend=True,
                              name = str(j)) , row=2, col=2)    
fig.update_xaxes(title_text="Bloques horarios", row = 2, col = 2)
fig.update_yaxes(title_text="Costos marginales [$/MWh]", row = 2, col = 2)

for j in rdf_V.columns:    
    fig.add_trace( go.Scatter(x = rdf_V.index, 
                              y = rdf_V[j], 
                              mode="lines", 
                              showlegend=True,
                              name = str(j)) , row=2, col=3)    
fig.update_xaxes(title_text="Bloques horarios", row = 2, col = 3)
fig.update_yaxes(title_text="Costos marginales [$/MWh]", row = 2, col = 3)

fig.show()        
        
  #%%

df_Costo_Operativo  =pd.DataFrame({"Costo_Operativo":rdf_Costo_Operativo},index=[Texto])

EnExcel_Resultados =     pd.ExcelWriter(Texto+".xlsx",engine='xlsxwriter')
df_Costo_Operativo.to_excel(EnExcel_Resultados,sheet_name="rdf_Costo_Operativo")
rdf_pr.to_excel(EnExcel_Resultados,sheet_name="rdf_pr")
rdf_Ph.to_excel(EnExcel_Resultados,sheet_name="rdf_Ph")
rdf_Sp.to_excel(EnExcel_Resultados,sheet_name="rdf_Sp")
rdf_V.to_excel(EnExcel_Resultados,sheet_name="rdf_V")
rdf_Pt.to_excel(EnExcel_Resultados,sheet_name="rdf_Pt")
rdf_pr.to_excel(EnExcel_Resultados,sheet_name="rdf_pr")
EnExcel_Resultados.save()
EnExcel_Resultados.close()
      













