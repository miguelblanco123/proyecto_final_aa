# Estructura propuesta — Presentación Ejecutiva

Detección de Clientes Anómalos en E-commerce (DBSCAN vs. Isolation Forest)

> Guía de contenido para armar el deck. Tono: técnico pero digerible — cada slide explica la decisión y el "para qué", no el detalle de implementación (ese vive en el notebook / PDF técnico).

---

## 1. Portada

- Título del proyecto
- Universidad / materia / profesor
- Integrantes del equipo
- Fecha

---

## 2. Definición del proyecto (1 slide)

**Mensaje central:** ¿qué problema de negocio resolvemos y por qué importa?

- Contexto: tienda de e-commerce (UK, 2009-2011), dataset público Online Retail / Online Retail II.
- Pregunta de negocio: ¿qué clientes se comportan de forma atípica respecto al resto de la base?
- Por qué no es un problema supervisado: no existe una etiqueta de "cliente anómalo" — es aprendizaje no supervisado.
- Enfoque: comparar dos algoritmos (DBSCAN e Isolation Forest) y quedarnos con el más accionable.

*Evitar:* jerga de código, nombres de columnas técnicas — eso va en la sección de Datos.

---

## 3. Datos (2-3 slides)

**Slide 3.1 — De transacciones a perfil de cliente**
- De N transacciones a 1 fila por cliente (`CustomerID`).
- Cifra clave: 5,876 clientes, 8 variables continuas + 2 categóricas (cumple mínimos del proyecto).
- Diagrama simple: Transacciones → Agregación por cliente → Dataset de features.

**Slide 3.2 — Qué describe a un cliente**
- Tabla resumen de las variables construidas (agrupadas por tipo), en lenguaje de negocio:
  - Lealtad/antigüedad: Permanencia
  - Frecuencia: Compras
  - Valor de compra: Canasta_Prom, Ticket_Prom, Precio_Prom, Precio_Max
  - Variedad: Productos Distintos
  - Riesgo/calidad: Pct_Devoluciones
  - Geografía: Pais Principal, Paises Distintos
- 1-2 hallazgos del EDA que enganchen (ej. "91% de los clientes compran principalmente en UK", "la mitad de los clientes solo compra 3 veces en 2 años").

**Slide 3.3 — Preparación de los datos (breve, sin fórmulas)**
- Corrección de sesgo en las distribuciones (transformación log/potencia).
- Codificación simple de país (2 banderas en vez de 40+ categorías).
- Estandarización de variables numéricas.
- Mensaje: "los datos crudos no estaban listos para comparar distancias entre clientes; este paso lo resuelve."

---

## 4. Modelo (2-3 slides)

**Slide 4.1 — Dos formas de definir "anómalo"**
- DBSCAN (visto en clase): agrupa por densidad; lo que no cae en ningún grupo = ruido/anomalía.
- Isolation Forest: aísla directamente los puntos "raros" con árboles aleatorios, sin necesidad de densidad.
- Analogía simple: DBSCAN dibuja "islas" de clientes parecidos y todo lo que quede fuera es sospechoso; Isolation Forest busca directamente a los clientes que se separan del resto con pocas preguntas.

**Slide 4.2 — Por qué comparamos dos modelos**
- DBSCAN es el modelo de clustering visto en clase — punto de partida natural.
- Isolation Forest se agrega y se justifica porque es el estándar de la industria para detección de anomalías tabulares, no depende de densidad, y sí permite clasificar clientes nuevos (`predict()`), a diferencia de DBSCAN.

**Slide 4.3 — Cómo se afinaron (alto nivel)**
- Optimización automática de hiperparámetros con Optuna (20 combinaciones por modelo), maximizando *silhouette score*.
- No es prueba y error manual: es una búsqueda sistemática y reproducible.

---

## 5. Evaluación (2 slides)

**Slide 5.1 — Comparación cuantitativa**
- Tabla/gráfica de barras: Silhouette Score y % de clientes marcados como anomalía, por modelo.
- DBSCAN: silhouette ≈ 0.08, ~39% marcado como anomalía.
- Isolation Forest: silhouette ≈ 0.58, ~1.3% marcado como anomalía.
- Mensaje: "39% no es un grupo accionable; 1.3% sí lo es."

**Slide 5.2 — Quiénes son los clientes atípicos (Isolation Forest)**
- Perfil del cliente anómalo: precio promedio y máximo elevados, ticket promedio alto, tasa de devoluciones alta.
- Visual: boxplots o proyección PCA con los clientes anómalos resaltados.
- Traducir a negocio: "clientes que compran artículos de alto valor y/o devuelven con frecuencia — candidatos a revisión de riesgo o a trato preferencial VIP, según el caso."

---

## 6. Conclusión (1 slide)

- Isolation Forest es el modelo recomendado para detección de anomalías en este dataset: mejor separación, porcentaje accionable, perfil interpretable.
- DBSCAN aporta valor como herramienta de segmentación general, no como detector de anomalías.
- Siguientes pasos: validar el perfil con el equipo de negocio, monitorear el modelo en producción, reentrenar periódicamente.

---

## 7. Cierre / Preguntas

- Agradecimientos
- Referencias: dataset UCI, repositorio del proyecto

---

### Notas de diseño para el deck

- 12-15 slides en total, máximo texto por slide: 1 idea + 1 visual.
- Reusar las gráficas ya generadas en `notebooks/notebook.ipynb` (PCA, boxplots, comparación de barras) — no rehacerlas desde cero.
- Guardar el detalle metodológico (fórmulas, código, hiperparámetros exactos) para el documento PDF técnico, no para el deck.
