# Perfiles de investigación EUR/USD

En GitHub Actions, abrí el workflow **Forex M15 research and Telegram**,
seleccioná la rama `main` y usá `mode=train`. Elegí un perfil y dejá
`timeframe=auto` y `expiry_candles=0` para aplicar sus valores.

| Perfil | Temporalidad | Expiración binaria por defecto | Confianza nominal | Entropía PPO | Coste de omitir un setup | Ventana de tendencia |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| scalping | M5, admite M1 | 1 vela, admite 1 a 3 | M5 binario 57%. M1 y spot 54%, admiten 52–54% | M5 binario 0,012. M1 y spot 0,03 | 0,20 | 4 barras M5 / 6 barras M1 |
| intraday | M15 | 1 vela | 60% | 0,02 | 0,05 | 8 barras |
| daily | H1 | 2 velas | 63% | 0,02 | 0,05 | 12 barras |
| swing | H4 | 2 velas | 65% | 0,02 | 0,05 | 16 barras |

`trading_profile=auto` elige el perfil de la temporalidad explícita. Con ambos
campos en automático se aplica intraday. El formulario rechaza combinaciones
incompatibles, como scalping con M15. `expiry_candles` acepta un entero y usa
cero como selector del preset. Forex spot conserva su contrato con
`expiry_candles=1` y evalúa posiciones de mercado sin vencimiento binario.
Los inputs de confianza, escala de momentum, entropía y coste neutral en `auto` aplican el preset.

## Histórico separado para scalping

Por defecto el runner obtiene el ZIP anual M1 de EUR/USD de 2025 mediante el
formulario oficial de HistData. Comprueba el SHA256 registrado en
`datasets/eurusd/manifest.json` y reutiliza el ZIP verificado desde la caché
de Actions. El año fijo permite repetir el experimento. El script separa el
Parquet M1 o M5, las ausencias observadas y la auditoría de origen. El manifiesto
del modelo identifica el ZIP original y la transformación aplicada.
[Histórico oficial M1 de 2025](https://www.histdata.com/download-free-forex-historical-data/?/ascii/1-minute-bar-quotes/eurusd/2025)

- Para M1 se usan las velas de un minuto publicadas por HistData, convertidas de EST fijo a UTC.
- Para M5 el camino automático agrega grupos completos de cinco velas M1 reales. Descarta grupos incompletos y registra los huecos. M5 también admite cargar OHLCV nativo de cinco minutos directamente.
- Las variables de Actions `FOREX_M1_DATA_DIR` y `FOREX_M5_DATA_DIR` permiten usar directorios estáticos propios. El cargador usa `config/closures_m1.json` o `config/closures_m5.json` si existe y exige documentar cualquier hueco. Separá las resoluciones por directorio.
- Intraday, daily y swing usan la fuente estática M15 existente. H1/H4 se forman solo con grupos completos de esa fuente.

El archivo verificado produjo 372.024 velas M1 y 73.775 grupos M5 completos.

La ingesta comprueba la resolución declarada, los timestamps, la coherencia
OHLCV y los hashes. M15 carece de los cierres intermedios para construir M1/M5.
El pipeline exige fuentes con granularidad suficiente. Las expiraciones
conservan los límites de los segmentos y de cada partición.

M1 usa una ventana GARCH máxima de 3.072 observaciones y reajusta cada
480 barras, equivalente a la cadencia de 96 barras M5 en minutos observados.
Esta ventana más amplia resolvió dos ajustes consecutivos que no convergían
con la ventana de 2.016 observaciones de la configuración base. En M1/M5, si un
reajuste GARCH aislado falla por falta de convergencia, se
mantienen los últimos parámetros válidos durante un intervalo y se continúa la
recursión causal. El fallo queda registrado en `garch_retained_refits`. El
entrenamiento aborta si falta un ajuste previo o si fallan dos consecutivos.

## Medir la frecuencia

El objetivo de frecuencia ejecutada es de 1,5 a 3 contratos por hora
observable para M5 binario y de 5 a 10 para M1 binario. El momentum usa 6
barras en M1 y 4 en M5. Con `drift_scale=auto`, M1 usa 1,8 fijo. M5 binario
interpola entre 1,4 y 2,0
según la relación entre la media de retornos recientes y su error estándar.
Usa el mínimo hasta una relación de 1 y el máximo desde 2,5. Ambos valores se
calculan con velas cerradas, antes de proyectar. En M5 binario se puede
seleccionar `auto` o `2.0` para el techo. P05/P95 y las
probabilidades direccionales se obtienen de la misma simulación de 10.000
trayectorias. Intraday, daily y swing conservan sus presets.

Con payout 0,80, el acierto de equilibrio es 55,56%. M5 binario exige
probabilidad direccional superior al 57% nominal y aplica Wilson con
`z=2.24`. La ventaja esperada calculada con ese límite debe superar el 1%
de la apuesta, lo que requiere un límite inferior mayor al 56,11% con ese
payout. M1 conserva
`z=1.96` y margen adicional cero. Wilson mide el error de muestreo de Monte
Carlo bajo el GBM. La validación mide por separado el acierto observado.

Los logs de validación y los reportes JSON y Markdown separan tres etapas.
Se cuenta la dirección MC en cada ventana con entrada y vencimiento válidos,
las propuestas que superan la confianza nominal y las que superan el filtro
de Wilson y payout. Los contratos ejecutados por PPO se muestran aparte.
Las tasas usan las mismas horas de toda la validación en segmentos operables,
incluso si el stop termina la ejecución de la política antes del final.
Las propuestas son oportunidades por vela. La cota de contratos por hora
incorpora también el vencimiento sin solapamiento.

## Alineación y gate de Scalping M5 binario

En el formulario, elegí `mode=train`, `trading_profile=scalping`,
`timeframe=M5`, `trading_mode=binary` y `expiry_candles=1` para reproducir
la prueba de una vela. Usá `auto` para confianza, drift y entropía. El preset
fija el umbral nominal en 57% y la entropía en 0,012. Se acepta un
`ent_coef` explícito de 0,01 a 0,012.

Una solicitud contraria a un setup que pasa los filtros recibe −1,25 en
la recompensa de PPO. Una operación alineada ganadora recibe una base de
+1,75 antes de aplicar costes de actividad, penalizaciones por drawdown
y stop. El saldo conserva la liquidación por payout y el filtro mantiene
bloqueada la dirección contraria.

El gate exige que el checkpoint entrenado cumpla todos los objetivos.

| Métrica de validación | Requisito |
| --- | ---: |
| Acierto sin reembolsos | ≥ 56% |
| Profit Factor | > 1,20 |
| Contratos ejecutados por hora observable | 1,5–3,0 |
| Contratos ejecutados sobre todos los setups elegibles | ≥ 30% |

La selección prioriza los checkpoints entrenados que pasan el gate y
desempata con el score habitual. Si ninguno pasa, Actions conserva los
reportes JSON, Markdown y CSV y marca el job como fallido antes de publicar
el bundle M5. El manifiesto registra `validation_gate_passed`. La conversión
usa todos los setups de validación, incluidos los posteriores a un stop,
y el reporte desglosa coincidencias, solicitudes contrarias y abstenciones
entre los setups que PPO llegó a evaluar.

Los bundles de investigación tienen nombres como
`forex-research-bundle-v4-scalping-M1-binary-e1`. El nombre y el manifiesto
distinguen perfil, temporalidad, tipo de mercado y expiración. La ejecución
original del v4 M15 conserva su propio artefacto. La partición de validación
respeta el corte histórico congelado y Telegram selecciona sus bundles por
su contrato de producción.
[Bundle v4 M15 original](https://github.com/Asriel550/Forex/actions/runs/42331180138)


## Referencia de la PR #11 con el histórico de 2025

Se completaron ingesta, features y 16.384 pasos PPO con semilla 42, umbral
nominal 55% y expiración de una vela en ambos perfiles de scalping.

| Validación simulada | M1 | M5 |
| --- | ---: | ---: |
| Setups antes de PPO | 344 | 37 |
| Contratos del checkpoint seleccionado | 4 | 8 |
| Contratos por hora observable | 0,0032 | 0,0066 |
| Cota de contratos por hora con el filtro | 0,2786 | 0,0307 |

La meta de 5–10 contratos por hora queda fuera de la capacidad del filtro en
esta muestra. El incentivo de recompensa solo actúa sobre setups elegibles.
M1 registró tres reajustes GARCH aislados retenidos con el estado causal.
El holdout quedó reservado y se conservaron separados los pesos de prueba.


## Referencia de sensibilidad de la PR #12

Dos corridas con el histórico de 2025, semilla 42, 16.384 pasos PPO, umbral
nominal 54% y una vela de expiración. La evaluación se limitó a validación y el
holdout quedó reservado.

| Métrica de validación | M1 | M5 |
| --- | ---: | ---: |
| Direcciones MC por hora antes del filtro | 59,99 | 12,00 |
| Propuestas por hora tras confianza nominal | 25,39 | 5,31 |
| Setups por hora tras Wilson y payout | 21,55 | 4,76 |
| Setups elegibles en toda la fase | 26.606 | 5.740 |
| Contratos ejecutados por PPO por hora | 0,290 | 0,222 |
| Acierto sin reembolsos | 53,47% | 51,15% |
| PnL simulado | −US$130 | −US$206 |
| Drawdown máximo | 20,04% | 20,60% |

En M5 la cota aumentó de 0,031 a 4,76 setups por hora sobre la misma fase. En
ambas corridas el stop de drawdown se activó y el acierto quedó por debajo del
55,56% necesario con payout 0,80. El ajuste amplió el conjunto de candidatos
para investigación. La frecuencia de ejecución de 5–10 por hora sigue pendiente.

## Comparación de drift de la PR #13

Se reutilizaron las volatilidades GARCH causales del histórico M5 de 2025
para aislar el efecto del drift. Con la misma validación y payout 0,80,
se evaluó la dirección de mayor probabilidad en cada setup elegible.

| Filtro y drift | Setups | Setups/h | Acierto resuelto del motor | Profit Factor del motor |
| --- | ---: | ---: | ---: | ---: |
| PR #12, 54%, fijo 2,2 | 5.740 | 4,76 | 48,46% | 0,752 |
| PR #13, 58%, fijo 2,2 | 5.315 | 4,41 | 48,28% | 0,747 |
| PR #13, 58%, adaptativo 1,4–2,2 | 4.112 | 3,41 | 47,91% | 0,736 |

El drift adaptativo reduce el número de setups en un 22,6% frente a la
escala fija con el mismo filtro. En esta validación, el acierto del motor
también baja. Las dos configuraciones de la PR #13 admitieron los mismos
candidatos después del umbral nominal y del filtro económico. El holdout
quedó reservado.

Con semilla 42 y 100.000 pasos solicitados por corrida, se seleccionaron
los checkpoints con el criterio habitual de validación ajustado por riesgo.
Ambas corridas usaron 58%, el mismo Wilson y el mismo presupuesto PPO.

| Checkpoint seleccionado en validación | Fijo 2,2 | Adaptativo 1,4–2,2 |
| --- | ---: | ---: |
| Paso | 75.000 | 50.000 |
| Contratos | 53 | 63 |
| Contratos/h | 0,044 | 0,052 |
| Acierto sin reembolsos | 58,49% | 55,74% |
| Profit Factor | 1,127 | 1,007 |
| PnL simulado | +US$28 | +US$2 |
| Drawdown máximo | 6,20% | 6,79% |
| Objetivos cumplidos | 1 de 3 | 0 de 3 |

A payout 0,80 y apuesta fija, PF superior a 1,2 requiere más de 60% de
acierto en los contratos resueltos. El drift adaptativo reduce propuestas
y deja una cota de 3,41 setups/h. La frecuencia efectivamente ejecutada
por PPO y la calidad conjunta siguen por debajo de los objetivos.

## Prueba de alineación de la PR #14

Se usó la fuente M5 de la corrida de la PR #13 y se reproyectó el drift
causal con techo de 2,0. Con semilla 42, 100.000 pasos solicitados y
vencimiento de una vela, el candidato diagnóstico seleccionado tuvo 219
contratos, 50,94% de acierto sin reembolsos, PF 0,831 y PnL de −US$176.
El stop de drawdown del 20% terminó la simulación. Hasta ese momento PPO
coincidió en 219 de 275 setups, se abstuvo en 56 y pidió cero direcciones
contrarias. Sobre la fase completa fueron 0,182 contratos por hora y
4,94% de conversión sobre 4.429 setups. El gate quedó rechazado y el
holdout reservado.

## Evolución genética de Scalping M5

En **Run workflow**, seleccioná `main` y `mode=evolution`. Los valores
`trading_profile=auto`, `timeframe=auto`, `trading_mode=binary` y
`expiry_candles=0` resuelven Scalping M5 binario con una vela de vencimiento.
Elegí `population_size` entre 4 y 128 y `generations` entre 1 y 20. Los
valores por defecto son 128 individuos y 10 ciclos posteriores a la
generación inicial, para un total de 11 generaciones evaluadas.

El genoma puntúa todas las entradas con vencimiento M5 consecutivo.
Un vector de pesos decide si hay contrato. Otro ajusta el margen
direccional `mc_prob_up - mc_prob_down`. El signo resultante decide
CALL o PUT si la magnitud supera el umbral de activación del genoma.
Un margen igual o inferior al umbral produce NEUTRO. Las observaciones
incluyen GARCH, Monte Carlo, indicadores técnicos y nueve variables
de contexto.

- **H1 y H4.** Diferencia de EMA8 y EMA24 del cierre, dividida por
  la media exponencial de 14 rangos. Cada vela se forma con 12 o
  48 velas M5 completas en UTC. Se usa el último cierre disponible
  al decidir, con vigencia menor a 60 o 240 minutos respectivamente.
- **Volatilidad M5.** Rango de la vela dividido por el cierre y raíz
  de la suma de los cuadrados de 12 retornos log consecutivos. Un
  hueco reinicia la ventana de volatilidad realizada.
- **Sesión.** Dos indicadores marcan si la entrada o su vencimiento
  tocan 16:30–18:00 en `America/New_York` o 08:00–10:00 en
  `Asia/Tokyo`. Se incluye el inicio y se excluye el fin. La hora
  local de NY incorpora el horario de verano.

La ausencia de un sesgo H1/H4 vigente o de volatilidad realizada se
codifica con valor cero e indicador de disponibilidad propio. La dirección,
confianza de Monte Carlo, volatilidad y sesión pueden influir mediante
los pesos de la política. Las reglas de caja, huecos, reembolsos y stop
al 20% se aplican al liquidar. Las horas incluyen todo el tramo observable.

Cada individuo recibe un puntaje finito. El acierto sin reembolsos
se pondera por `min(1, f/1.2)`, donde `f` es la frecuencia ejecutada
sobre todas las horas observables. La distancia por debajo de 1,2 pesa
el doble que por encima de 2,5. Se resta el drawdown como fracción y un
castigo exponencial desde el 12%, con un máximo de 0,50 para ese término.
El acierto también tiene una penalización suave por quedar por debajo
del objetivo efectivo del payout. Con stake fijo y payout del 80%,
un PF de 1,20 equivale al 60% de acierto sin reembolsos. La aptitud
usa ese objetivo para penalizar el acierto y el gate comprueba PF por
separado.

El primer 75% de validación cronológica se divide en cuatro bloques
consecutivos de tamaño similar. Cada bloque empieza con saldo propio y
liquida sus contratos dentro de sus límites. Cada transición toma 25%
del puntaje anterior y 75% del posterior y descuenta el 25% del exceso
de deterioro sobre 0,05 puntos. La aptitud pondera el tramo completo
al 25%, la media de transiciones al 37,5% y la peor al 37,5%.
La varianza de acierto se registra para diagnóstico. Los cuatro bloques
pertenecen a la búsqueda y la confirmación se aplica al campeón congelado.

En búsqueda también se resta 0,15 veces la distancia de la proporción
de señales descartadas por el umbral al rango de 20% a 30%. Se mide
sobre las señales que propondría el mismo genoma con umbral cero,
antes de aplicar el stop. El reporte separa las señales admitidas por
el umbral de las que quedan sin ejecutar por el stop. La frecuencia del
gate usa los contratos ejecutados y las horas observables del tramo.

El grupo de padres conserva el tamaño del 20% superior redondeado hacia
arriba. La aptitud ordena candidatos y el PnL por hora desempata.
El mejor individuo elegible pasa intacto a la generación siguiente
como padre de referencia. Entre los demás padres, quienes terminan en
positivo con drawdown menor al 15% en los cuatro bloques también pasan
como élites. Para cada descendiente ordinario, el motor elige dos padres
distintos del grupo y mezcla sus pesos, sesgos y umbral con un coeficiente
aleatorio entre 0,25 y 0,75. Después aplica la mutación ordinaria.
Si el grupo contiene un solo padre, el descendiente parte de una copia.

Cada generación reserva aproximadamente el 18,75% de la población para
saltos directos desde el mejor padre, con redondeo al entero más próximo
y un mínimo de un individuo. Con 128 individuos se reservan 24 plazas.
El salto aplica ruido gaussiano a los pesos y al umbral, con desviación
estándar de tres veces la mutación ordinaria y un mínimo de 0,35. Estos
descendientes pueden reproducirse o quedar como campeones si superan la
aptitud global del padre, igualan o mejoran su puntaje económico en cada
bloque de búsqueda y conservan los criterios del gate que ya cumplía.
El CSV distingue estas mutaciones y su elegibilidad, con puntajes por
bloque para revisar la comparación.

La evaluación de búsqueda completa transporta el saldo entre bloques
para medir el riesgo de todo el tramo. Tras una vela de purga, se evalúa
una sola vez al campeón preseleccionado en confirmación. El gate final
exige consistencia en los cuatro bloques y, en búsqueda completa y
confirmación, acierto sin reembolsos ≥56%, PF >1,20, 1,2–2,5 contratos por
hora y drawdown <15%. El holdout histórico queda reservado.

### Continuidad del linaje

La primera corrida parte de la generación 21. El individuo cero
hereda los pesos de entrada, el sesgo y el normalizador del campeón
Gen 20 de la PR #18. El ZIP original y su SHA256 se conservan en
`config/evolution_m5_seed_gen20.json`. Los pesos de contexto y el
ajuste direccional del fundador parten en cero. Se ajusta la
normalización del contexto exclusivamente en train durante la migración.
Cada corrida reevalúa al fundador con la aptitud y los bloques vigentes
antes de generar la población inicial. El resto de la población nace
por mutación del fundador.
[Corrida de la PR #18](https://github.com/Asriel550/Forex/actions/runs/42421933732)

Actions guarda el campeón de búsqueda y el estado evolutivo en
`forex-evolution-lineage-v4` antes de aplicar el gate de validación.
La siguiente ejecución manual en la rama por defecto recupera el
checkpoint disponible más reciente por fecha de creación del mismo workflow, incluidos los
de corridas rechazadas por el gate. Continúa después de la última
generación evaluada, aunque el campeón se haya seleccionado antes.
Al recuperar un checkpoint v3 o v4, el umbral del fundador empieza
en cero para reproducir sus decisiones. El umbral muta en los
descendientes y se guarda junto con los pesos en el formato v5.
Al faltar un checkpoint vigente, Actions parte del fundador archivado.
La retención solicitada del checkpoint es de 90 días. Para recuperar
una copia archivada localmente, el motor acepta `--seed-bundle lineage.zip`.

El artefacto `forex-reports-*` contiene JSON, CSV por individuo y
Markdown con entradas evaluables, descartes por el umbral y contratos CALL/PUT. Cada generación
publica su propio candidato `bundle_scalping_gen_X.zip` con genoma,
normalizador y manifiesto con SHA256. Los artefactos de evolución se
usan exclusivamente para investigación M5.

### Reproducción histórica

La corrida local usó 64 individuos, 20 ciclos y semilla 42 sobre
el histórico M5 de 2025 de la PR #18. Evaluó las generaciones 21 a 41
y seleccionó la 39 en búsqueda. Pasaron al genoma 10.854 entradas
evaluables en búsqueda y 3.614 en confirmación.

| Confirmación | PR #18 Gen 20 | Gen 39 |
| --- | ---: | ---: |
| Contratos ejecutados | 107 | 398 |
| Acierto sin reembolsos | 52,34% | 53,47% |
| Profit Factor | 0,878 | 0,919 |
| Contratos por hora | 0,355 | 1,322 |
| Drawdown máximo | 9,75% | 20,78% |
| PnL simulado | −US$62 | −US$146 |

El campeón activó el stop y quedó rechazado por el gate. La prueba
de continuidad heredó sus pesos y comenzó en la generación 42.
Se reutilizó la confirmación histórica para el diagnóstico y el
holdout quedó reservado.

### Ablación direccional de la Gen 273

El workflow manual **Ablación direccional Scalping M5** descarga el
checkpoint de la #84 y comprueba su SHA256. La admisión se reconstruye
con la Gen 268 completa en cada ventana. El cruce y la mutación actúan
sobre los genes direccionales. Monte Carlo puro participa como control
de la primera población con los pesos de ajuste en cero.

La aptitud suma al puntaje económico una medida direccional calculada
sobre todas las velas elegibles de los cuatro bloques internos. La
medida combina dos veces el diferencial de payoff por vela frente a
Monte Carlo y 0,10 veces la diferencia de Spearman. Para agregar los
bloques se ponderan por igual su media y su mínimo. En empate exacto
del puntaje direccional se toma el lado Monte Carlo o el lado original
del fundador si Monte Carlo también empata.

Tras seleccionar al campeón y guardar el checkpoint, el workflow
carga 2026 y reproduce marzo con la máscara congelada. El reporte
compara el campeón con la Gen 268 y Monte Carlo sobre las mismas
entradas. Marzo se usa como diagnóstico retrospectivo. El artefacto
experimental requiere el checkpoint de entradas y se guarda fuera
del linaje estándar. ATR y M15 quedan para pruebas separadas.

### Comparación M15 de las Gen 294 a 314

El workflow manual **Comparación direccional M15 en Scalping M5**
continúa desde el checkpoint experimental de la Gen 291 y mantiene
la admisión de la Gen 268. Los tres brazos usan la misma semilla,
128 individuos, 20 ciclos y los mismos bloques internos de búsqueda,
preparados una vez en el mismo runner.

El control conserva la dirección original. El brazo con feature
añade a la dirección la diferencia EMA 8–24 de velas M15 cerradas,
dividida por su propio rango medio y recortada a ±3. El coeficiente
heredable se limita a ±6. El brazo con ancla añade un coeficiente
fijo de +12 cuando el GARCH M5 supera el percentil 67 del tramo de
entrenamiento. Con ese sesgo direccional todavía pueden coexistir
señales CALL y PUT. Las entradas conservan su máscara.

Después de congelar a los tres campeones, se prepara marzo una sola vez
y se comprueba el hash de la misma
máscara de marzo. El reporte también reproduce a cada campeón con
su término M15 apagado y cuenta cambios de lado y aciertos en las
primeras 101 admisiones. La comparación de marzo es retrospectiva.

### Benchmark supervisado de dirección M5

El workflow manual **Benchmark supervisado de dirección Scalping M5**
verifica los SHA256 de los checkpoints Gen 268 y Gen 306 archivados
en `config/m5-direction-benchmark/`. Provienen de la
[corrida #84](https://github.com/Asriel550/Forex/actions/runs/42448607949)
y de la [ablación M15](https://github.com/Asriel550/Forex/actions/runs/42455178389).
La Gen 268 determina todas las admisiones y la Gen 306 aporta una
referencia de dirección.

Una regresión logística multinomial regularizada aprende PUT, empate
y CALL al cierre M5 siguiente, con las observaciones causales ya
disponibles. El ajuste usa datos de 2025 hasta el 26 de noviembre a las
00:00 UTC. El resto de 2025 calibra por separado las probabilidades de
la logística y de Monte Carlo. Los contratos liquidan dentro de cada
ventana. Antes de cargar 2026 se fijan ambos calibradores y el
percentil 67 de GARCH del ajuste.

El reporte compara Monte Carlo original, Monte Carlo recalibrado,
logística calibrada y Gen 306 sobre cada admisión, incluidas las que
hubiera cortado el stop. El retorno por admisión se expresa en unidades
de stake y el intervalo del diferencial pareado remuestrea días UTC.
El informe presenta los regímenes de volatilidad por separado y
calcula intervalos para grupos con al menos diez días. Enero, febrero
y marzo son retrospectivos. El tramo nuevo de abril termina en la
frontera protegida del 7 de abril a las 15:15 UTC, antes de construir
features. Es una observación corta para la prueba de repetibilidad.

La comparación mide señal direccional. Un replay con abstención,
stop y gate requiere una ventaja repetible en bloques posteriores.

### Ablación multiescala de las Gen 315 a 335

El workflow manual **Ablación multiescala de dirección Scalping M5**
conserva el checkpoint Gen 268 para la admisión y proyecta las variables
tácticas del control Gen 294 en la primera de ocho neuronas tanh. Las
tres ramas tienen el mismo genoma y activan M5, M5+M15 o M5+M15+H1.
El puntaje de salida determina CALL o PUT.

M15 aporta volatilidad realizada de ocho velas y ATR relativo de
Wilder de 14 velas. H1 aporta la inclinación relativa de la EMA 24
y el rango horario medio relativo. Ambos bloques se agregan desde M1
completo y usan el último cierre disponible en el instante de decisión.
Los valores ausentes toman el centro de la escala ajustada en marzo.

Las semillas 42, 43 y 44 recorren 21 poblaciones de 128 individuos
por rama. Las 2.859 aperturas de marzo se archivan como un bitmap de
la cuadrícula UTC de M5. Se comprueba su hash histórico y se valida
que cada apertura sigue siendo elegible en el histórico de entrada.
Enero y febrero aportan el historial causal de los indicadores. La
selección y la escala usan exclusivamente las admisiones de marzo.

El fitness pondera 45 % el retorno global por admisión y 55 % el peor
sub-bloque. Resta 0,20 veces la desviación de log PF, regularizado con
cinco stakes a cada lado. Se usan cuatro bloques cronológicos, cada
uno separado por el percentil 67 de GARCH de marzo cuando ambos
estratos tienen al menos 64 admisiones. Tienen prioridad los genomas
con retorno positivo y mejora frente a Monte Carlo en cada sub-bloque.

El artefacto de marzo contiene el hash y los pesos de un campeón por
rama. El workflow lo publica antes de preparar el diagnóstico conocido
de abril. El replay exige el hash de las 278 admisiones de la PR #27
y termina en la frontera protegida del 7 de abril a las 15:15 UTC.
Los dos reportes puntúan el retorno direccional antes del stop.

### Replay operativo Gen 333 y extensión hasta la Gen 355

El workflow manual **Replay operativo Gen 333 y extensión Gen 336–355**
verifica el sello de la Gen 333, semilla 42, y ambas máscaras archivadas.
Sobre sus propuestas aplica el filtro M5 de Monte Carlo al lado elegido
por el genoma. Usa probabilidad nominal superior al 57 %, Wilson con
z de 2,24 y ventaja conservadora superior al 1 % del stake. Para este
diagnóstico, también excluye la volatilidad superior al percentil 67
fijado en marzo y admite hasta dos entradas por hora móvil.

El stop de cuenta suspende nuevas operaciones al alcanzar un drawdown
del 20 % desde el pico tras una liquidación. La corrida histórica
liquida al vencimiento con stake fijo y payout archivado. El informe
separa los descartes del gate de las entradas que quedan pendientes
por stop, y el retorno sobre el stake total ejecutado del PnL en dólares.

Para salidas por contrato, el comando de replay admite
`--exit-quotes`, `--stop-loss` y `--take-profit` juntos. El CSV
requiere `entry_utc`, `expiry_utc`, `quote_utc`, `side` (CALL/PUT)
y `sell_proceeds_usd` netos para el mismo contrato y stake del replay.
Los umbrales se expresan como fracción de stake. Se liquida en la primera
cotización ejecutable que cruza el umbral. El histórico M1 del
subyacente solo aporta liquidación por vencimiento.

Tras el replay, las semillas 42, 43 y 44 arrancan de la misma Gen 333.
Cada una evalúa 20 poblaciones de 128 individuos sobre el snapshot de
marzo y la escala sellada. Conserva el fitness multirrégimen y guarda
un genoma ganador con su hash por semilla. Abril termina en el corte
protegido y sirve exclusivamente para el replay de la Gen 333.

### Gate adaptativo elástico de la Gen 354

La ablación ejecuta el genoma sellado Gen 354/43 sobre la matriz
canónica de marzo y el diagnóstico de abril. Compara el gate estático,
convicción p85 y la variante elástica. El contraste histórico con un
techo de volatilidad queda documentado en la corrida de la PR #30.
La convicción es el valor absoluto de tanh aplicado al puntaje de
salida. El p85 usa las 500 propuestas anteriores, incluidos los
descartes. Los empates con el corte se descartan. Febrero aporta las
500 propuestas de arranque y el estado continúa de marzo a abril.

Para ampliar la admisión, la convicción media debe superar 0,80 y las
últimas 500 propuestas resueltas deben alcanzar 95 % de acierto en
cada uno de tres bloques cronológicos. Se observan también las propuestas descartadas cuando
vence el contrato y se excluyen reembolsos. La política pasa de p85
a p60, p30 y finalmente p0 tras cada 100 liquidaciones adicionales
mientras mantiene esa evidencia. Si la pierde, regresa al percentil inicial.
El cupo horario y el stop de cuenta se aplican tras la decisión de
admisión. El replay usa liquidación al vencimiento.

### Evolución del gate de convicción, Gen 355 a 375

El workflow compara p85, p80 y p75, con 128 individuos, semillas
42, 43 y 44 y la Gen 354/43 como fundador común. El ranking prioriza
PF estrictamente superior a 1,20 en marzo. Entre quienes pasan el PF,
busca acercarse a 1,2 contratos/h. El margen monetario sobre el PF
objetivo desempata. Cada candidato usa su propio historial móvil y
pasa por el cupo horario y el stop de cuenta.

El workflow publica los genomas y el arranque antes de cargar abril.
El replay cruza cada campeón y el fundador con los tres percentiles.
El informe guarda el rango y número de cortes observados para verificar
que la convicción exigida cambia durante el replay. Se reporta también
la capacidad de las admisiones congeladas con el cupo horario.
