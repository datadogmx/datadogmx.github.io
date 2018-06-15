---
layout: post
title: "PCA with pyspark"
featured-img: shane-rounce-205187
categories: [pca, pyspark, datascience]
mathjax: true
---


# Análisis de componentes principales (PCA)

---

* Principal component analysis
 - Spectral decomposition of covariance matrix.
 - Constant probability density contours and confidence hyper-ellipses.
 - Maximum likelihood estimators.
* Simulating multivariate normal random vectors.


# Principal component analysis

# Análisis de componentes principales

## Bases matemáticas 

    En esta sección vamos a dar algunos de los elementos matemáticos necesarios para desarrollar y entender PCA

### Estadística

Vamos a representar a la muestra de una población como $ X $ un ejemplo sería: <br/>
$$ X = [1,2,4,6,.....,98]$$ 
donde $X_4=6$ $\leftarrow$ un valor dentro de nuestra muestra

### Media

La media de una dimensión se define como:
$$\overline{X} = \frac{
                            \sum_{i=1}^{n}X_i
                       }{n}$$
                       
### Desviación estándar

" La distancia promedio de la media a los puntos de los datos "

$$ s = \sqrt{ \frac{
                 \sum_{i=1}^{n}(X_i - \overline{X} )^2
            }{n-1} } $$

### Varianza


$$ s^2 = \frac{
                 \sum_{i=1}^{n}(X_i - \overline{X} )^2
            }{n-1} $$
            
### Covarianza

Esta fórmula nos sirve para saber qué tanto una dimensión varía con respecto a otra:

$$ cov(X,Y) = \frac{
                 \sum_{i=1}^{n}(X_i - \overline{X} )(Y_i - \overline{Y} )
            }{n-1} $$
### Matriz de covarianza

$$ C^{nxn} = (C_{i,j},C_{j,i} = cov(Dim_i,Dim_j)) $$

Donde C^{nxn} es una matriz con n columnas y n renglones

## Álgebra lineal
Para PCA necesitamos saber acerca de $eigenvalores$ y $eigenvectores$

### Eigenvectors y Eigenvalues

Sea $A$ una matriz cuadrada con eigenvalores ($\lambda$) y eigenvectores ($v$), siguiendo la siguiente regla:

$$ A\overrightarrow{v} = \lambda \overrightarrow{v}$$

## Otros temas importantes

- Multiplicidad algebraica y geométrica
- Singular Value Decomposition (SVD)
- Multiplicador de Lagrange

## Desarrollo del método

Nustro objetivo será buscar una transformación que mejor conserve la información y reducir la dimensionalidad (complejidad) de nuestros datos
<img src='static/pca1.gif'>

El nuevo punto mapeado
$$ U_1^T X $$

Donde $U_1$ es el vector que usamos para transformar los datos. <br/>
Un enfoque de conservar la mayor información posible es maximizando la varianza, por lo tanto tenemos que maximizar:

$$ var(U_1^T X) $$

Desagregando la fórmula:


$$ var(U_1^T X) = U_1^TSU_1 $$

Donde $S$ es la matriz de covarianza de $X$

Bajo esta condición cualquier $U_1$ podria maximizar la expresión por lo tanto necesitamos poner una restricción:

$$ U_1^TU_1 = 1$$

Usando los multiplicadores de Lagrange obtenemos la siguiente expresión:
$$  SU_1 = \lambda U_1$$

La expresión es la expresión de los vectores y valores característicos de $S$.

Al obtener los valores característicos de $S$ los ordenamos de mayor a menor, y elegimos el vector correspondiente al valor característico mayor como nuestro primer componente, y así con el resto de los siguientes componentes

$$ \lambda_1 > \lambda_2 > \lambda_3 > ...  > \lambda_d $$
$$     u_1   >   u_2     >     u_3   > ... > u_d $$

## Ejemplo  [Análisis de componentes principales para un conjunto de vinos]

### Cargando las librerías


```python
run MvaUtils.py
```


```python
%matplotlib inline
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pyspark import SparkContext
from IPython.display import display, HTML
from pyspark.sql import SQLContext
from pyspark.ml.feature import PCA
from pyspark.ml.linalg import Vectors
from pyspark.sql import Column as c
from pyspark.sql.functions import array, udf, lit, col as c
import pyspark.sql.functions as f
pd.set_option('max_colwidth',100)
plt.style.use('seaborn-paper')
```

Cargando el spark Context


```python

try:
    sc = SparkContext('local[*]')
except:
    sc = SparkContext.getOrCreate('local[*]')

sqlContext = SQLContext(sc)
```

Leyendo los dataframes


```python
whiteWinnePath = 'data/winequalityWhite.csv'
redWinnePath = 'data/winequalityRed.csv'

whiteWinneDF = sqlContext.createDataFrame(pd.read_csv(whiteWinnePath)).withColumn('type',lit(0))

redWinneDF = sqlContext.createDataFrame(pd.read_csv(redWinnePath)).withColumn('type',lit(1))
        
redWinneDF.printSchema()
```

    root
     |-- fixedAcidity: double (nullable = true)
     |-- volatileAcidity: double (nullable = true)
     |-- citricAcid: double (nullable = true)
     |-- residualSugar: double (nullable = true)
     |-- chlorides: double (nullable = true)
     |-- freeSulfurDioxide: double (nullable = true)
     |-- totalSulfurdioxide: double (nullable = true)
     |-- density: double (nullable = true)
     |-- pH: double (nullable = true)
     |-- sulphates: double (nullable = true)
     |-- alcohol: double (nullable = true)
     |-- quality: long (nullable = true)
     |-- type: integer (nullable = false)
    


Dividiendo conjuntos de entrenamiento y prueba


```python
whiteTrainingDF, whiteTestingDF = whiteWinneDF.randomSplit([0.7,0.3])

redTrainingDF, redTestingDF = redWinneDF.randomSplit([0.7,0.3])
        
trainingDF = whiteTrainingDF.union(redTrainingDF)

testingDF = whiteTestingDF.union(redTestingDF)
```

Preparando el dataframe para PCA


```python
idCol = ['type']

features = [column for column in redWinneDF.columns if column not in idCol]

p = len(features)

meanVector = trainingDF.describe().where(c('summary')==lit('mean'))\
                       .toPandas().as_matrix()[0][1:p+1]
        
labeledVectorsDF = trainingDF.select(features+['type']).rdd\
                             .map(lambda x:(Vectors.dense(x[0:p]-Vectors.dense(meanVector)),x[p]))\
                             .toDF(['features','type'])

labeledVectorsDF.limit(5).toPandas()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>features</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[-2.20879048249, -0.166617096277, 0.240927517074, -3.97387089667, -0.0298993170302, -6.850297422...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[-2.20879048249, 0.213382903723, -0.179072482926, 2.82612910333, -0.0238993170302, 4.14970257766...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[-2.10879048249, -0.00661709627671, -0.0990724829258, -3.87387089667, -0.0288993170302, -12.8502...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[-2.10879048249, -0.00661709627671, -0.0990724829258, -3.87387089667, -0.0288993170302, -12.8502...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[-2.10879048249, -0.00661709627671, -0.0990724829258, -3.87387089667, -0.0288993170302, -12.8502...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Generando PCA


```python
k=2
pcaModel = PCA(k=k, inputCol="features", outputCol="featuresPCA").fit(labeledVectorsDF)
transformedFeaturesDF = pcaModel.transform(labeledVectorsDF)

plt.figure(figsize=(7,7))
eigenvalues = [float(value) for value in pcaModel.explainedVariance]
plt.bar(list(range(1,len(eigenvalues)+1)),eigenvalues)
pd.DataFrame(eigenvalues,columns=['explained variance'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>explained variance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.953103</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.040951</td>
    </tr>
  </tbody>
</table>
</div>




![png](output_17_1.png)


Obteniendo un dataframe con los puntos mapeados


```python
schema = ['pc'+str(k+1) for k in range(0,k)]

transformedWhiteWinneDF = transformedFeaturesDF.where(c('type') == lit(0))\
                                               .select(['featuresPCA']).rdd\
                                               .map(lambda r: [float(r[0].values[i]) for i in range (0,k)])\
                                               .toDF(schema)

transformedRedWinneDF = transformedFeaturesDF.where(c('type') == lit(1))\
                                             .select(['featuresPCA']).rdd\
                                             .map(lambda r: [float(r[0].values[i]) for i in range (0,k)])\
                                             .toDF(schema)

projectedCanonicalsPD = pd.DataFrame([c for c in pcaModel.pc.toArray()],columns=schema)
```

### Graficando la transformacion


```python
col_1,col_2='pc1','pc2'
alpha=0.05
freedomDegrees=2
plt.figure(figsize=(16,16))
plt.axis([-180,150,-50,75])

scatterPlot(plt,transformedWhiteWinneDF,col_1,col_2,'Gray')
summaryWhiteWinnePD = getProbabilityDensityContour(plt,transformedWhiteWinneDF,\
                            [col_1,col_2],alpha,freedomDegrees,\
                             color='Gray',name='Transformed White Winne')

scatterPlot(plt,transformedRedWinneDF,col_1,col_2,'Red')
summaryRedWinnePD = getProbabilityDensityContour(plt,transformedRedWinneDF,\
                            [col_1,col_2],alpha,freedomDegrees,\
                             color='Red',name='Transformed Red Winne')

plt.xlabel(col_1+' '+str(eigenvalues[0]*100)+' %')
plt.ylabel(col_2+' '+str(eigenvalues[1]*100)+' %')

plt.show()
display(summaryWhiteWinnePD)
display(summaryRedWinnePD)
```


![png](output_21_0.png)



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Summary</th>
      <th>Transformed White Winne</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mean</td>
      <td>[22.5400472872, -0.536387450207]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Covariance matrix</td>
      <td>[[1901.07537348, 50.5506184042], [50.5506184042, 167.013239864]]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EigenValues</td>
      <td>[1902.54775274, 165.540860604]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EigenVectors</td>
      <td>[[-0.999576083602, 0.0291144824798], [-0.0291144824798, -0.999576083602]]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Confidence</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Chi-squared critical value</td>
      <td>5.99146</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Summary</th>
      <th>Transformed Red Winne</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mean</td>
      <td>[-71.4078633066, 1.69929908468]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Covariance matrix</td>
      <td>[[1135.99106569, -0.454987253641], [-0.454987253641, 63.010475229]]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EigenValues</td>
      <td>[1135.99125862, 63.010282296]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EigenVectors</td>
      <td>[[-0.999999910095, -0.000424040411327], [0.000424040411327, -0.999999910095]]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Confidence</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Chi-squared critical value</td>
      <td>5.99146</td>
    </tr>
  </tbody>
</table>
</div>


## PCA con la muestra normalizada

### Normalizando datos


```python
idCol = ['type']

features = [column for column in redWinneDF.columns if column not in idCol]

aggExpresions = [f.mean(c(colName)).alias('mean'+colName) for colName in features]+\
                [f.stddev(c(colName)).alias('stddev'+colName) for colName in features]


stdExpresions = [((c(colName)-c('mean'+colName))/c('stddev'+colName))\
                 .alias('std'+colName) for colName in features]

statisticsDF = trainingDF.select(aggExpresions)

stdTrainingDF = trainingDF.crossJoin(f.broadcast(statisticsDF))\
                              .select([c(idCol[0])]+stdExpresions)
    
stdTrainingDF.limit(5).toPandas()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>type</th>
      <th>stdfixedAcidity</th>
      <th>stdvolatileAcidity</th>
      <th>stdcitricAcid</th>
      <th>stdresidualSugar</th>
      <th>stdchlorides</th>
      <th>stdfreeSulfurDioxide</th>
      <th>stdtotalSulfurdioxide</th>
      <th>stddensity</th>
      <th>stdpH</th>
      <th>stdsulphates</th>
      <th>stdalcohol</th>
      <th>stdquality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>-1.734762</td>
      <td>-1.021185</td>
      <td>1.697573</td>
      <td>-0.830817</td>
      <td>-0.846972</td>
      <td>-0.389219</td>
      <td>-0.029746</td>
      <td>-1.353841</td>
      <td>1.651094</td>
      <td>-0.953022</td>
      <td>0.264639</td>
      <td>1.343810</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>-1.734762</td>
      <td>1.307810</td>
      <td>-1.261743</td>
      <td>0.590859</td>
      <td>-0.677007</td>
      <td>0.235777</td>
      <td>0.842710</td>
      <td>-0.956633</td>
      <td>1.965899</td>
      <td>-0.138760</td>
      <td>1.705992</td>
      <td>2.488192</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>-1.656223</td>
      <td>-0.040556</td>
      <td>-0.698064</td>
      <td>-0.809910</td>
      <td>-0.818644</td>
      <td>-0.730126</td>
      <td>-0.492682</td>
      <td>-1.784150</td>
      <td>1.839977</td>
      <td>-1.020877</td>
      <td>1.705992</td>
      <td>1.343810</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>-1.656223</td>
      <td>-0.040556</td>
      <td>-0.698064</td>
      <td>-0.809910</td>
      <td>-0.818644</td>
      <td>-0.730126</td>
      <td>-0.492682</td>
      <td>-1.784150</td>
      <td>1.839977</td>
      <td>-1.020877</td>
      <td>1.705992</td>
      <td>1.343810</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>-1.656223</td>
      <td>-0.040556</td>
      <td>-0.698064</td>
      <td>-0.809910</td>
      <td>-0.818644</td>
      <td>-0.730126</td>
      <td>-0.492682</td>
      <td>-1.784150</td>
      <td>1.839977</td>
      <td>-1.020877</td>
      <td>1.705992</td>
      <td>1.343810</td>
    </tr>
  </tbody>
</table>
</div>



Realizando la transformación


```python
stdSchema = [column for column in stdTrainingDF.columns if column not in idCol]
p = len(stdSchema)

meanVector = stdTrainingDF.describe().where(c('summary')==lit('mean'))\
                       .toPandas().as_matrix()[0][1:p+1]
        
labeledVectorsDF = stdTrainingDF.select(stdSchema+['type']).rdd\
                             .map(lambda x:(Vectors.dense(x[0:p]-Vectors.dense(meanVector)),x[p]))\
                             .toDF(['features','type'])

labeledVectorsDF.limit(5).toPandas()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>features</th>
      <th>type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>[-1.97468290092, -1.02118504116, 1.69757300084, -0.83081705125, -0.846971956383, -0.389218932102...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>[-1.97468290092, 1.3078095477, -1.26174301674, 0.590858714117, -0.677007146378, 0.235777033644, ...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>[-1.89614388931, -0.0405557405853, -0.698063775293, -0.8099100547, -0.818644488049, -0.730125822...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>[-1.89614388931, -0.0405557405853, -0.698063775293, -0.8099100547, -0.818644488049, -0.730125822...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>[-1.89614388931, -0.0405557405853, -0.698063775293, -0.8099100547, -0.818644488049, -0.730125822...</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Obteniendo componentes de los datos normalizados


```python
k=4
pcaModel = PCA(k=k, inputCol="features", outputCol="featuresPCA").fit(labeledVectorsDF)
transformedFeaturesDF = pcaModel.transform(labeledVectorsDF)

plt.figure(figsize=(7,7))
eigenvalues = [float(value) for value in pcaModel.explainedVariance]
plt.bar(list(range(1,len(eigenvalues)+1)),eigenvalues)
pd.DataFrame(eigenvalues,columns=['explained variance'])
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>explained variance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.252457</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.222515</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.135722</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.089695</td>
    </tr>
  </tbody>
</table>
</div>




![png](output_28_1.png)


Obteniendo un dataframe con los puntos mapeados


```python
schema = ['pc'+str(k+1) for k in range(0,k)]

transformedWhiteWinneDF = transformedFeaturesDF.where(c('type') == lit(0))\
                                               .select(['featuresPCA']).rdd\
                                               .map(lambda r: [float(r[0].values[i]) for i in range (0,k)])\
                                               .toDF(schema)

transformedRedWinneDF = transformedFeaturesDF.where(c('type') == lit(1))\
                                             .select(['featuresPCA']).rdd\
                                             .map(lambda r: [float(r[0].values[i]) for i in range (0,k)])\
                                             .toDF(schema)

projectedCanonicalsPD = pd.DataFrame([c for c in pcaModel.pc.toArray()],columns=schema)
```

### Biplot


```python
col_1,col_2='pc1','pc2'
alpha=0.05
freedomDegrees=2
plt.figure(figsize=(16,16))
plt.axis([-7,5,-6,6])

plotProjectedBase(plt,['pc1','pc2'],projectedCanonicalsPD,features)

scatterPlot(plt,transformedWhiteWinneDF,col_1,col_2,'Gray')
summaryWhiteWinnePD = getProbabilityDensityContour(plt,transformedWhiteWinneDF,\
                            [col_1,col_2],alpha,freedomDegrees,\
                             color='Gray',name='Transformed White Winne')

scatterPlot(plt,transformedRedWinneDF,col_1,col_2,'Red')
summaryRedWinnePD = getProbabilityDensityContour(plt,transformedRedWinneDF,\
                            [col_1,col_2],alpha,freedomDegrees,\
                             color='Red',name='Transformed Red Winne')


plt.xlabel(col_1+' '+str(eigenvalues[0]*100)+' %')
plt.ylabel(col_2+' '+str(eigenvalues[1]*100)+' %')

plt.show()
display(summaryWhiteWinnePD)
display(summaryRedWinnePD)
```

    /opt/conda/lib/python3.6/site-packages/matplotlib/cbook/deprecation.py:107: MatplotlibDeprecationWarning: Adding an axes using the same arguments as a previous axes currently reuses the earlier instance.  In a future version, a new instance will always be created and returned.  Meanwhile, this warning can be suppressed, and the future behavior ensured, by passing a unique label to each axes instance.
      warnings.warn(message, mplDeprecation, stacklevel=1)



![png](output_32_1.png)



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Summary</th>
      <th>Transformed White Winne</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mean</td>
      <td>[0.887516064024, -0.23334104858]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Covariance matrix</td>
      <td>[[0.866278445996, 0.717277172469], [0.717277172469, 2.82962298827]]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EigenValues</td>
      <td>[3.06374962734, 0.632151806925]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EigenVectors</td>
      <td>[[0.310298404615, -0.950639206057], [0.950639206057, 0.310298404615]]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Confidence</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Chi-squared critical value</td>
      <td>5.99146</td>
    </tr>
  </tbody>
</table>
</div>



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Summary</th>
      <th>Transformed Red Winne</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mean</td>
      <td>[-2.54969647808, 0.486481732887]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Covariance matrix</td>
      <td>[[0.901572863795, -0.391490544277], [-0.391490544277, 1.77299176953]]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>EigenValues</td>
      <td>[1.92303616734, 0.751528465991]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>EigenVectors</td>
      <td>[[-0.357879909174, -0.933767621311], [0.933767621311, -0.357879909174]]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Confidence</td>
      <td>0.95</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Chi-squared critical value</td>
      <td>5.99146</td>
    </tr>
  </tbody>
</table>
</div>