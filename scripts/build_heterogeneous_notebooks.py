"""Genera 02B y adapta 03/04.

Debe ejecutarse desde la raíz del proyecto:
    python scripts/build_heterogeneous_notebooks.py
"""
import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(True)}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": text.splitlines(True)}


cells = []
cells.append(md("""# 02B — MLP heterogéneo multientrada con Keras

Continuación directa del notebook 02. Comparamos, en un único split estratificado 80/20 (semilla 42), la mejor arquitectura MLP encontrada allí (`128-64-32`) con una arquitectura multientrada que respeta la naturaleza financiera de cada variable.

**Regla de selección (se decide aquí una sola vez):** gana el modelo con mayor **PR-AUC de validación**. Esta métrica es la usada por la búsqueda de arquitectura del MLP de 02, es apropiada para una clase positiva minoritaria y no depende del umbral. El conjunto de test no interviene: el 20% es exclusivamente validación. Los umbrales de coste se estiman también solo en validación y se guardan para 03/04."""))
cells.append(code("""from __future__ import annotations
import os, random, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, f1_score, matthews_corrcoef, precision_score, recall_score, roc_auc_score)

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers
except ImportError as exc:
    raise ImportError('Instala TensorFlow y pydot: pip install tensorflow pydot graphviz') from exc

import importlib
import utils.heterogeneous_mlp as heterogeneous_utils

# Útil durante el desarrollo interactivo: si el módulo cambió después de
# arrancar el kernel, se recarga para no conservar una versión antigua.
heterogeneous_utils = importlib.reload(heterogeneous_utils)
Log1pNonNegative = heterogeneous_utils.Log1pNonNegative
dataframe_to_keras_inputs = heterogeneous_utils.dataframe_to_keras_inputs
save_json = heterogeneous_utils.save_json
stable_hash = heterogeneous_utils.stable_hash

warnings.filterwarnings('ignore')
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED); np.random.seed(SEED); tf.keras.utils.set_random_seed(SEED)
try: tf.config.experimental.enable_op_determinism()
except Exception: pass

OUTPUT_DIR=Path('outputs'); MODELS_DIR=OUTPUT_DIR/'models'; OBJECTS_DIR=OUTPUT_DIR/'objects'; PLOTS_DIR=OUTPUT_DIR/'plots'
for p in (MODELS_DIR, OBJECTS_DIR, PLOTS_DIR): p.mkdir(parents=True, exist_ok=True)
pd.set_option('display.max_columns', 100)
print('TensorFlow:', tf.__version__, '| seed:', SEED)"""))
cells.append(md("""## 1. Datos procesados y particionado común

Se carga `outputs/preprocessed_train.csv`: el **DataFrame procesado por 01**, con ausentes imputados, extremos financieros recortados y variables derivadas calculadas. No se usa el CSV original porque rompería la continuidad con el baseline de 02.

El CSV procesado conserva columnas auxiliares, pero se reconstruye exactamente la regla `FINAL_FEATURES` de 01: se excluyen las versiones `_log1p` y los indicadores binarios derivados redundantes. Deben quedar 16 variables. Ambos modelos se entrenan desde cero con los mismos índices 80/20 estratificados."""))
cells.append(code("""data_path = OUTPUT_DIR/'preprocessed_train.csv'
if not data_path.exists(): raise FileNotFoundError('Ejecuta primero 01_EDA_Preprocesado.ipynb')
df = pd.read_csv(data_path)
TARGET='SeriousDlqin2yrs'
LOG1P_AUXILIARY=[c for c in df.columns if c.endswith('_log1p')]
DERIVED_BINARY_AUXILIARY=['Has90DaysLate','HasAnyPastDue','HasDependents']
AUXILIARY_COLUMNS=LOG1P_AUXILIARY+DERIVED_BINARY_AUXILIARY
FEATURES=[c for c in df.columns if c != TARGET and c not in AUXILIARY_COLUMNS]
assert len(FEATURES)==16, f'Se esperaban las 16 FINAL_FEATURES de 01 y se obtuvieron {len(FEATURES)}: {FEATURES}'
X=df[FEATURES].copy(); y=df[TARGET].astype('int32')
train_idx, val_idx = train_test_split(np.arange(len(df)), test_size=.20, random_state=SEED, stratify=y)
X_train, X_val=X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
y_train, y_val=y.iloc[train_idx].copy(), y.iloc[val_idx].copy()
np.savez(OBJECTS_DIR/'mlp_80_20_split_seed42.npz', train_idx=train_idx, val_idx=val_idx)
assert set(train_idx).isdisjoint(set(val_idx)) and len(train_idx)+len(val_idx)==len(df)
print('Fuente procesada:',data_path)
print('FINAL_FEATURES:',len(FEATURES),FEATURES)
print('Train/validación:',X_train.shape,X_val.shape,'| prevalencia:',y_train.mean(),y_val.mean())"""))
cells.append(md("""## 2. Tipología funcional, no basada solo en `dtype`

- **Continuas financieras:** ratios e ingreso poseen alta cardinalidad y colas extremas. Se reciben en la escala procesada seleccionada por 01; la rama aplica `log1p` internamente y normaliza usando solo train, sin añadir las columnas auxiliares `_log1p`.
- **Edad:** es entera, pero representa una magnitud ordenada casi continua; se conserva numérica y normalizada. Convertirla en categoría perdería proximidad entre edades.
- **Conteos de mora:** 30–59, 60–89 y 90+ días comparten significado y vocabulario (número de episodios). Se aplica un **embedding compartido**: aquí sí hay equivalencia semántica del token `k` entre variables.
- **Otros conteos:** líneas, préstamos inmobiliarios, dependientes y total de retrasos tienen vocabularios distintos. Usan embeddings **independientes**; compartirlos impondría que, por ejemplo, “2 dependientes” equivalga a “2 hipotecas”.
- **Binarias:** los cuatro indicadores de ausencia seleccionados por 01 entran directamente. Los indicadores derivados `Has*` continúan excluidos por redundancia.
- **Nominales:** no hay categorías nominales originales en este dataset.
- **Ordinales:** tampoco hay categorías ordinales etiquetadas. La edad y los conteos sí tienen orden y se conserva mediante una vía numérica adicional para los conteos.
- **Derivadas:** `TotalPastDueEvents` y `CreditLinesPerRealEstateLoan`, incluidas por 01, se mantienen. Los conteos pasan simultáneamente por embedding y `log1p` numérico para conservar orden y permitir no linealidad."""))
cells.append(code("""CONTINUOUS=['RevolvingUtilizationOfUnsecuredLines','DebtRatio','MonthlyIncome','CreditLinesPerRealEstateLoan']
INTEGER_ORDERED=['age']
DELINQUENCY_COUNTS=['NumberOfTime30-59DaysPastDueNotWorse','NumberOfTime60-89DaysPastDueNotWorse','NumberOfTimes90DaysLate']
INDEPENDENT_COUNTS=['NumberOfOpenCreditLinesAndLoans','NumberRealEstateLoansOrLines','NumberOfDependents','TotalPastDueEvents']
BINARY=[c for c in FEATURES if c.endswith('_was_missing')]
USED=set(CONTINUOUS+INTEGER_ORDERED+DELINQUENCY_COUNTS+INDEPENDENT_COUNTS+BINARY)
EXCLUDED=[c for c in FEATURES if c not in USED]
assert not EXCLUDED and USED==set(FEATURES), f'Clasificación funcional incompleta: {EXCLUDED}'

assign={**{c:('continua','log1p + Normalization','continuous_input') for c in CONTINUOUS},
        **{c:('entera ordenada','Normalization','integer_input') for c in INTEGER_ORDERED},
        **{c:('conteo mora','embedding compartido + log1p','delinquency branch') for c in DELINQUENCY_COUNTS},
        **{c:('conteo','embedding independiente + log1p','independent embeddings') for c in INDEPENDENT_COUNTS},
        **{c:('binaria','entrada 0/1 directa','binary_input') for c in BINARY}}
rows=[]
for c in FEATURES:
    s=X[c]; typ,trans,branch=assign[c]
    rows.append({'variable':c,'dtype':str(s.dtype),'tipo_funcional':typ,'n_unicos':s.nunique(dropna=True),
      'min':s.min(),'max':s.max(),'ausentes_pct':100*s.isna().mean(),'p01':s.quantile(.01),'mediana':s.median(),
      'p99':s.quantile(.99),'transformacion':trans,'rama':branch})
variable_summary=pd.DataFrame(rows)
display(variable_summary)
variable_summary.to_csv(OUTPUT_DIR/'heterogeneous_variable_summary.csv',index=False)
print('Las 16 FINAL_FEATURES están clasificadas:',sorted(USED))
print('Auxiliares del CSV no usadas:',AUXILIARY_COLUMNS)"""))
cells.append(code("""fig, axes=plt.subplots(3,3,figsize=(15,10)); axes=axes.ravel()
for ax,c in zip(axes, CONTINUOUS+INTEGER_ORDERED+DELINQUENCY_COUNTS[:3]+INDEPENDENT_COUNTS[:1]):
    ax.hist(X[c].clip(upper=X[c].quantile(.995)),bins=40); ax.set_title(c,fontsize=8)
plt.tight_layout(); plt.show()"""))
cells.append(md("""## 3. Contrato de preprocesamiento

Las capas `Normalization` se adaptan **exclusivamente con train** y se serializan dentro de cada `.keras`. Los embeddings reciben enteros recortados al máximo observado en train; el contrato y sus máximos se guardan con una huella SHA-256. 03 y 04 verifican esa huella y los nombres de entrada antes de predecir."""))
cells.append(code("""embedding_cfg={c:{'max_token':int(np.ceil(X_train[c].max())),'input_dim':int(np.ceil(X_train[c].max()))+2,
                          'embedding_dim':min(4,max(2,int(np.ceil(np.log2(X_train[c].nunique()+1)))))}
               for c in DELINQUENCY_COUNTS+INDEPENDENT_COUNTS}
spec={'version':1,'features':FEATURES,'vector_branches':{'continuous_input':CONTINUOUS,'integer_input':INTEGER_ORDERED,
      'binary_input':BINARY,'count_numeric_input':DELINQUENCY_COUNTS+INDEPENDENT_COUNTS},
      'embedding_features':embedding_cfg,'shared_embedding_group':DELINQUENCY_COUNTS,'excluded_features':AUXILIARY_COLUMNS,
      'split':{'method':'train_test_split_stratified','test_size':.20,'random_state':SEED}}
spec_hash=stable_hash(spec)
print('Huella:',spec_hash); display(pd.DataFrame(embedding_cfg).T)"""))
cells.append(md("""## 4. Modelos Keras

El baseline replica la mejor arquitectura de 02: 128–64–32, BatchNorm, ReLU y dropout 0,25; permanece intacto para conservar la referencia.

El heterogéneo es deliberadamente más compacto para reducir sobreajuste: ramas de 16, 4, 8 y 12 neuronas, embeddings de hasta 4 dimensiones y una cabeza final de solo dos capas, 64–32. Después de concatenar se añade ruido gaussiano suave durante entrenamiento, regularización L2=5e-4 y dropout 0,35/0,25. El ruido se desactiva automáticamente en inferencia. Ambos modelos conservan optimizador, learning rate, batch, ponderación de clase, callbacks y split."""))
cells.append(code("""def norm_layer(name, data):
    layer=layers.Normalization(name=name); layer.adapt(np.asarray(data,dtype='float32')); return layer

def build_baseline(Xtr):
    inp=keras.Input((len(FEATURES),),name='baseline_input')
    x=norm_layer('baseline_normalization',Xtr[FEATURES])(inp)
    for units,drop in [(128,.25),(64,.25),(32,.125)]:
        x=layers.Dense(units,kernel_regularizer=regularizers.l2(1e-4))(x)
        x=layers.BatchNormalization()(x); x=layers.Activation('relu')(x); x=layers.Dropout(drop)(x)
    out=layers.Dense(1,activation='sigmoid',name='probability')(x)
    return keras.Model(inp,out,name='baseline_keras_128_64_32')

def build_heterogeneous(Xtr,spec):
    reps=[]; inputs=[]
    for name, cols, units in [('continuous_input',CONTINUOUS,16),('integer_input',INTEGER_ORDERED,4),('binary_input',BINARY,8)]:
        inp=keras.Input((len(cols),),name=name); inputs.append(inp)
        if name=='continuous_input':
            x=Log1pNonNegative(name='continuous_log1p')(inp)
            x=norm_layer(name+'_norm',np.log1p(Xtr[cols].clip(lower=0)))(x)
        elif name=='binary_input':
            x=inp
        else:
            x=norm_layer(name+'_norm',Xtr[cols])(inp)
        x=layers.Dense(units,activation='relu',name=name+'_dense')(x); reps.append(x)
    count_cols=DELINQUENCY_COUNTS+INDEPENDENT_COUNTS
    cin=keras.Input((len(count_cols),),name='count_numeric_input'); inputs.append(cin)
    clog=Log1pNonNegative(name='counts_log1p')(cin)
    clog=norm_layer('counts_log_norm',np.log1p(Xtr[count_cols].clip(lower=0)))(clog)
    reps.append(layers.Dense(12,activation='relu',name='count_numeric_dense')(clog))
    shared_dim=max(spec['embedding_features'][c]['input_dim'] for c in DELINQUENCY_COUNTS)
    shared_emb=layers.Embedding(shared_dim,3,name='shared_delinquency_embedding')
    for c in DELINQUENCY_COUNTS:
        inp=keras.Input((1,),dtype='int32',name='emb_'+c); inputs.append(inp)
        reps.append(layers.Flatten(name='flat_'+c)(shared_emb(inp)))
    for c in INDEPENDENT_COUNTS:
        cfg=spec['embedding_features'][c]; inp=keras.Input((1,),dtype='int32',name='emb_'+c); inputs.append(inp)
        emb=layers.Embedding(cfg['input_dim'],cfg['embedding_dim'],name='embedding_'+c)(inp)
        reps.append(layers.Flatten(name='flat_'+c)(emb))
    x=layers.Concatenate(name='concatenate_branches')(reps)
    x=layers.GaussianNoise(.03,name='representation_noise')(x)
    for units,drop in [(64,.35),(32,.25)]:
        x=layers.Dense(units,kernel_regularizer=regularizers.l2(5e-4))(x)
        x=layers.BatchNormalization()(x); x=layers.Activation('relu')(x); x=layers.Dropout(drop)(x)
    return keras.Model(inputs,layers.Dense(1,activation='sigmoid',name='probability')(x),name='heterogeneous_mlp')

baseline=build_baseline(X_train); heterogeneous=build_heterogeneous(X_train,spec)
for model in (baseline,heterogeneous):
    model.compile(optimizer=keras.optimizers.Adam(1e-3),loss='binary_crossentropy',
      metrics=[keras.metrics.AUC(name='roc_auc'),keras.metrics.AUC(curve='PR',name='pr_auc')])
baseline.summary(); heterogeneous.summary()"""))
cells.append(code("""for model,name in [(baseline,'baseline_architecture.png'),(heterogeneous,'heterogeneous_architecture.png')]:
    try:
        keras.utils.plot_model(model,to_file=str(PLOTS_DIR/name),show_shapes=True,show_layer_names=True,
                               show_dtype=True,expand_nested=True,dpi=120)
        display(Image(filename=str(PLOTS_DIR/name)))
    except Exception as exc:
        print('plot_model requiere pydot y Graphviz:',exc)"""))
cells.append(md("""## 5. Entrenamiento idéntico

Máximo 300 épocas; `EarlyStopping(patience=100)` restaura los mejores pesos según `val_pr_auc`; `ReduceLROnPlateau(patience=30)` reduce el learning rate a la mitad. La ponderación positiva reproduce el tratamiento del desbalance de 02."""))
cells.append(code("""class_weight={0:1.0,1:float((y_train==0).sum()/(y_train==1).sum())}
def callbacks(tag):
    return [keras.callbacks.EarlyStopping(monitor='val_pr_auc',mode='max',patience=100,restore_best_weights=True),
      keras.callbacks.ReduceLROnPlateau(monitor='val_pr_auc',mode='max',patience=30,factor=.5,min_lr=1e-6,verbose=1),
      keras.callbacks.ModelCheckpoint(MODELS_DIR/f'{tag}_best.keras',monitor='val_pr_auc',mode='max',save_best_only=True)]

baseline_inputs=X_train[FEATURES].astype('float32').to_numpy(); baseline_val=X_val[FEATURES].astype('float32').to_numpy()
hetero_inputs=dataframe_to_keras_inputs(X_train,spec); hetero_val=dataframe_to_keras_inputs(X_val,spec)
history_baseline=baseline.fit(baseline_inputs,y_train,validation_data=(baseline_val,y_val),epochs=300,batch_size=1024,
  class_weight=class_weight,callbacks=callbacks('baseline_keras'),verbose=2)
history_hetero=heterogeneous.fit(hetero_inputs,y_train,validation_data=(hetero_val,y_val),epochs=300,batch_size=1024,
  class_weight=class_weight,callbacks=callbacks('heterogeneous_mlp'),verbose=2)"""))
cells.append(md("""## 6. Evaluación, curvas y comparación

Se reportan las mismas métricas del notebook 02. Para cada escenario, el threshold que minimiza coste se busca únicamente en validación. La selección de arquitectura usa PR-AUC; los costes no se mezclan para evitar que dos escenarios produzcan dos modelos base distintos."""))
cells.append(code("""def best_threshold(y_true,p,fp_cost,fn_cost):
    best=None
    for t in np.linspace(.01,.99,199):
        pred=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y_true,pred,labels=[0,1]).ravel(); cost=fp_cost*fp+fn_cost*fn
        row=(cost,t)
        if best is None or row<best: best=row
    return float(best[1])
def metrics(name,p,t,fp_cost,fn_cost):
    pred=(p>=t).astype(int); tn,fp,fn,tp=confusion_matrix(y_val,pred,labels=[0,1]).ravel()
    return {'model':name,'threshold':t,'accuracy':accuracy_score(y_val,pred),'balanced_accuracy':balanced_accuracy_score(y_val,pred),
      'precision_1':precision_score(y_val,pred,zero_division=0),'recall_1':recall_score(y_val,pred,zero_division=0),
      'f1_1':f1_score(y_val,pred,zero_division=0),'mcc':matthews_corrcoef(y_val,pred),'roc_auc':roc_auc_score(y_val,p),
      'pr_auc':average_precision_score(y_val,p),'fp':int(fp),'fn':int(fn),'total_cost':int(fp_cost*fp+fn_cost*fn),
      'avg_cost':(fp_cost*fp+fn_cost*fn)/len(y_val)}

p_base=baseline.predict(baseline_val,batch_size=4096,verbose=0).ravel()
p_het=heterogeneous.predict(hetero_val,batch_size=4096,verbose=0).ravel()
scenarios={'cost_1_1':(1,1),'cost_1_10':(1,10)}; rows=[]; thresholds={}
for model_name,p in [('baseline_keras',p_base),('heterogeneous_mlp',p_het)]:
    thresholds[model_name]={}
    for scenario,(fpc,fnc) in scenarios.items():
        t=best_threshold(y_val.to_numpy(),p,fpc,fnc); thresholds[model_name][scenario]=t
        rows.append({'scenario':scenario,**metrics(model_name,p,t,fpc,fnc)})
comparison=pd.DataFrame(rows); display(comparison); comparison.to_csv(OUTPUT_DIR/'heterogeneous_vs_baseline_validation.csv',index=False)

fig,axes=plt.subplots(1,3,figsize=(16,4))
for hist,label in [(history_baseline,'baseline'),(history_hetero,'heterogéneo')]:
    axes[0].plot(hist.history['loss'],label=label); axes[0].plot(hist.history['val_loss'],'--',label='val '+label)
    axes[1].plot(hist.history['pr_auc'],label=label); axes[1].plot(hist.history['val_pr_auc'],'--',label='val '+label)
    axes[2].plot(hist.history['roc_auc'],label=label); axes[2].plot(hist.history['val_roc_auc'],'--',label='val '+label)
for ax,title in zip(axes,['Binary cross-entropy','PR-AUC','ROC-AUC']): ax.set_title(title); ax.legend(fontsize=7); ax.grid(alpha=.2)
plt.tight_layout(); plt.show()"""))
cells.append(md("""## 7. Selección persistida y trazabilidad

La comparación usa estrictamente `heterogeneous PR-AUC > baseline PR-AUC`; un empate conserva el baseline. Se guardan ambos modelos, pero 03 y 04 cargan el ganador indicado en el manifiesto. La decisión no vuelve a calcularse posteriormente."""))
cells.append(code("""scores={'baseline_keras':float(average_precision_score(y_val,p_base)),
        'heterogeneous_mlp':float(average_precision_score(y_val,p_het))}
winner='heterogeneous_mlp' if scores['heterogeneous_mlp']>scores['baseline_keras'] else 'baseline_keras'
baseline.save(MODELS_DIR/'baseline_keras_final.keras'); heterogeneous.save(MODELS_DIR/'heterogeneous_mlp_final.keras')
selected_file={'baseline_keras':'baseline_keras_final.keras','heterogeneous_mlp':'heterogeneous_mlp_final.keras'}[winner]
manifest={'version':1,'selection_source':'02B_MLP_Heterogeneo_Keras.ipynb','selection_set':'validation_20_percent',
 'selection_metric':'pr_auc','selection_rule':'max validation PR-AUC; ties keep baseline','random_state':SEED,
 'validation_scores':scores,'selected_model':{'name':winner,'model_family':'KerasMLP','model_file':selected_file},
 'candidate_models':{'baseline_keras':'baseline_keras_final.keras','heterogeneous_mlp':'heterogeneous_mlp_final.keras'},
 'thresholds':thresholds[winner],'scenarios':{'cost_1_1':{'fp_cost':1,'fn_cost':1,'output_file':'cs_produccion1.csv'},
 'cost_1_10':{'fp_cost':1,'fn_cost':10,'output_file':'cs_produccion2.csv'}},
 'preprocessing_spec':spec,'preprocessing_hash':spec_hash}
save_json(manifest,MODELS_DIR/'mlp_selection_manifest.json')
print('SELECCIONADO:',winner,'| PR-AUC:',scores[winner]); display(pd.DataFrame([scores]))"""))
cells.append(md("""## 8. Análisis crítico y conclusiones

Interprete la tabla ejecutada, no solo la pérdida. Una mejora de PR-AUC indica mejor ranking de la minoría en validación; compruebe además si baja el coste en ambos escenarios y si el cambio supera la variabilidad esperable de un único split. Si gana el heterogéneo, la mejora es compatible con el tratamiento especializado de colas, orden y conteos, pero no prueba causalidad de una rama concreta. Si no gana, la muestra puede no contener categorías nominales suficientes para que los embeddings aporten valor, y el baseline denso puede capturar ya las interacciones.

Limitaciones: una sola partición tiene mayor varianza que CV (decisión solicitada); el conjunto del 20% es validación, no test final; los embeddings de conteos recortan valores de producción por encima del máximo de train; y la arquitectura no se afinó mediante grid para evitar sesgo de búsqueda. 03 y 04 mostrarán el ganador y la PR-AUC que motivó la decisión."""))

nb={"cells":cells,"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
    "language_info":{"name":"python","version":"3"}},"nbformat":4,"nbformat_minor":5}
Path('02B_MLP_Heterogeneo_Keras.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')

# Adaptación modular de 03: se reemplazan solo las celdas de carga/modelo.
p3=Path('03_XAI_Auditoria.ipynb'); n3=json.loads(p3.read_text(encoding='utf-8'))
n3['cells'][0]=md("""# 03 — XAI y auditoría del mejor MLP validado

Este notebook carga la decisión única tomada por `02B_MLP_Heterogeneo_Keras.ipynb`. Puede auditar el baseline Keras o el MLP heterogéneo sin volver a seleccionar. El criterio es mayor PR-AUC sobre el 20% de validación (semilla 42); nunca se consulta test.""")
n3['cells'][3]=code("""import importlib
import utils.heterogeneous_mlp as heterogeneous_utils
heterogeneous_utils = importlib.reload(heterogeneous_utils)
load_selection_manifest = heterogeneous_utils.load_selection_manifest
load_keras_model_from_manifest = heterogeneous_utils.load_keras_model_from_manifest
predict_selected_proba = heterogeneous_utils.predict_selected_proba
verify_runtime_contract = heterogeneous_utils.verify_runtime_contract
required_files=[OUTPUT_DIR/'preprocessed_train.csv',MODELS_DIR/'mlp_selection_manifest.json']
for path in required_files:
    if not path.exists(): raise FileNotFoundError(f'Falta {path}. Ejecuta 01, 02 y 02B.')
train_df=pd.read_csv(OUTPUT_DIR/'preprocessed_train.csv'); manifest=load_selection_manifest()
TARGET='SeriousDlqin2yrs'; FEATURES=manifest['preprocessing_spec']['features']
X=train_df[FEATURES].astype(float); y=train_df[TARGET].astype(int)
audit_model=load_keras_model_from_manifest(manifest); verify_runtime_contract(audit_model,X,manifest)
model_metadata={'scenarios':manifest['scenarios'],'selected_models':{s:{'model_family':'KerasMLP',
 'model_file':manifest['selected_model']['model_file'],'threshold':manifest['thresholds'][s]} for s in manifest['scenarios']}}
print('Modelo seleccionado una sola vez:',manifest['selected_model']['name'])
print('Criterio:',manifest['selection_rule'],manifest['validation_scores'])""")
n3['cells'][4]=code("""def predict_model_score(model, model_family, X_array, batch_size=4096):
    frame=X_array if isinstance(X_array,pd.DataFrame) else pd.DataFrame(X_array,columns=FEATURES)
    return predict_selected_proba(model,frame,manifest,batch_size)
def predict_selected_scenario(scenario,X_raw):
    selected=model_metadata['selected_models'][scenario]
    score=predict_selected_proba(audit_model,X_raw[FEATURES],manifest)
    return score,(score>=selected['threshold']).astype(int),selected
class IdentityScaler:
    def transform(self,z): return np.asarray(z,dtype=float)
scaler=IdentityScaler(); X_scaled=scaler.transform(X)""")
n3['cells'][7]=code("""AUDIT_SCENARIO='cost_1_10'
audit_score=scenario_predictions[AUDIT_SCENARIO]['score']; audit_pred=scenario_predictions[AUDIT_SCENARIO]['pred']
audit_selected=scenario_predictions[AUDIT_SCENARIO]['selected']; audit_family='KerasMLP'
audit_threshold=audit_selected['threshold']; audit_checkpoint={'model_type':manifest['selected_model']['name']}
print('Escenario:',AUDIT_SCENARIO,'| modelo:',manifest['selected_model']['name'],'| threshold:',audit_threshold)
print('Selección motivada por PR-AUC validación:',manifest['validation_scores'])""")
p3.write_text(json.dumps(n3,ensure_ascii=False,indent=1),encoding='utf-8')

# Adaptación de 04: carga el mismo manifiesto y modelo; no re-selecciona.
p4=Path('04_Generacion_Predicciones.ipynb'); n4=json.loads(p4.read_text(encoding='utf-8'))
n4['cells'][0]=md("""# 04 — Predicciones con el MLP seleccionado en validación

Reutiliza exactamente la decisión persistida por 02B (mayor PR-AUC en validación 20%, semilla 42). Este notebook no evalúa ni selecciona modelos.""")
n4['cells'][3]=code("""import importlib
import utils.heterogeneous_mlp as heterogeneous_utils
heterogeneous_utils = importlib.reload(heterogeneous_utils)
load_selection_manifest = heterogeneous_utils.load_selection_manifest
load_keras_model_from_manifest = heterogeneous_utils.load_keras_model_from_manifest
predict_selected_proba = heterogeneous_utils.predict_selected_proba
verify_runtime_contract = heterogeneous_utils.verify_runtime_contract
required_files=[OUTPUT_DIR/'preprocessed_prod.csv',MODELS_DIR/'mlp_selection_manifest.json']
for path in required_files:
    if not path.exists(): raise FileNotFoundError(f'Falta {path}. Ejecuta 01, 02 y 02B.')
prod_df=pd.read_csv(OUTPUT_DIR/'preprocessed_prod.csv'); manifest=load_selection_manifest()
TARGET='SeriousDlqin2yrs'; FEATURES=manifest['preprocessing_spec']['features']
selected_model=load_keras_model_from_manifest(manifest); verify_runtime_contract(selected_model,prod_df,manifest)
MODEL_LOAD_STRATEGY=('reconstrucción modular + carga de pesos' if manifest['selected_model']['name']=='heterogeneous_mlp'
                     else 'deserialización Keras estándar')
model_metadata={'scenarios':manifest['scenarios'],'selected_models':{s:{'model_family':'KerasMLP',
 'model_file':manifest['selected_model']['model_file'],'threshold':manifest['thresholds'][s]} for s in manifest['scenarios']}}
print('Modelo fijado por 02B:',manifest['selected_model']['name'])
print('Estrategia de carga:',MODEL_LOAD_STRATEGY)
print('PR-AUC validación:',manifest['validation_scores'],'| criterio:',manifest['selection_rule'])""")
n4['cells'][5]=md("""## 2. Carga modular y contrato

Para el MLP heterogéneo, la arquitectura se reconstruye desde el módulo común y se cargan únicamente los pesos del archivo `.keras`. De este modo, producción no depende de deserializar `Lambda` ni clases personalizadas almacenadas en el artefacto. Para el baseline se mantiene la carga Keras estándar. En ambos casos se verifican la huella del preprocesamiento, las columnas y los nombres de entrada antes de predecir.""")
n4['cells'][6]=code("""def predict_for_scenario(scenario):
    selected=model_metadata['selected_models'][scenario]
    score=predict_selected_proba(selected_model,prod_df[FEATURES],manifest)
    pred=(score>=selected['threshold']).astype(int)
    return score,pred,selected""")
# El bloque antiguo de logística dependía del scaler de 02; se conserva la entrega principal y se elimina esa comparación lateral.
src=''.join(n4['cells'][8]['source']); marker='# Predicciones adicionales de la regresi'
if marker in src: src=src.split(marker)[0]
src += """
# Exportaciones comparativas de regresión logística entrenada en 02.
# Se mantienen separadas de la selección MLP: no intervienen en la elección
# baseline Keras vs. heterogéneo ni modifican sus thresholds.
logreg_required = [
    OBJECTS_DIR/'final_model_scaler.joblib',
    MODELS_DIR/'final_logistic_regression.joblib',
    MODELS_DIR/'model_metadata.joblib',
]
for path in logreg_required:
    if not path.exists():
        raise FileNotFoundError(f'Falta {path}. Ejecuta el notebook 02 para exportar la regresión logística.')

logreg_scaler = joblib.load(OBJECTS_DIR/'final_model_scaler.joblib')
logreg_model = joblib.load(MODELS_DIR/'final_logistic_regression.joblib')
legacy_metadata = joblib.load(MODELS_DIR/'model_metadata.joblib')
logreg_thresholds = legacy_metadata['logreg_thresholds']
X_prod_logreg = logreg_scaler.transform(prod_df[FEATURES].astype(float))
logreg_score = logreg_model.predict_proba(X_prod_logreg)[:, 1]
logreg_output_files = {
    'cost_1_1':'cs_produccion_logistica1.csv',
    'cost_1_10':'cs_produccion_logistica2.csv',
}
logreg_details=[]
for scenario, output_name in logreg_output_files.items():
    threshold=float(logreg_thresholds[scenario])
    pred=(logreg_score>=threshold).astype(int)
    output=original_prod.copy(); output[TARGET]=pred
    output.to_csv(PRED_DIR/output_name,index=False)
    logreg_details.append(pd.DataFrame({
        'row_id':np.arange(len(pred)),'scenario':scenario,
        'model_family':'LogisticRegression','model_file':'final_logistic_regression.joblib',
        'threshold':threshold,'score_class_1':logreg_score,'prediction':pred,
    }))
    print('Guardado adicional logístico:',PRED_DIR/output_name)
pd.concat(logreg_details,ignore_index=True).to_csv(
    PRED_DIR/'production_prediction_details_logistic.csv',index=False
)
"""
n4['cells'][8]=code(src)
n4['cells'][9]=code("""# Comprobaciones de formato de los dos ficheros generados por el modelo seleccionado.
for scenario, params in model_metadata['scenarios'].items():
    path = PRED_DIR / params['output_file']
    df_check = pd.read_csv(path)
    assert df_check.shape[0] == original_prod.shape[0], 'El número de filas debe coincidir con producción.'
    assert list(df_check.columns) == list(original_prod.columns), 'Las columnas deben coincidir con producción original.'
    assert df_check[TARGET].isin([0, 1]).all(), 'El target predicho debe ser binario 0/1.'
    print(path.name, 'OK', df_check.shape)

for output_name in logreg_output_files.values():
    path=PRED_DIR/output_name
    df_check=pd.read_csv(path)
    assert df_check.shape[0]==original_prod.shape[0]
    assert list(df_check.columns)==list(original_prod.columns)
    assert df_check[TARGET].isin([0,1]).all()
    print(path.name,'OK',df_check.shape)""")
n4['cells'][14]=md("""## 5. Ficheros finales

Los ficheros generados con el único modelo seleccionado en validación son:

- `outputs/predictions/cs_produccion1.csv`
- `outputs/predictions/cs_produccion2.csv`
- `outputs/predictions/production_prediction_details.csv`

Como comparación adicional de 02 también se conservan:

- `outputs/predictions/cs_produccion_logistica1.csv`
- `outputs/predictions/cs_produccion_logistica2.csv`
- `outputs/predictions/production_prediction_details_logistic.csv`

Estas exportaciones no participan en la selección entre los dos MLP; reutilizan el modelo, scaler y thresholds logísticos guardados por 02.""")
p4.write_text(json.dumps(n4,ensure_ascii=False,indent=1),encoding='utf-8')
print('Notebooks generados y adaptados.')
