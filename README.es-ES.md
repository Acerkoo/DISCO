

# Decodificación Consciente de Problemas Desactualizados para Preguntas de Razonamiento sobre Conocimiento Editado

Hallazgos ACL 2024: Decodificación Consciente de Problemas Desactualizados para Preguntas de Razonamiento sobre Conocimiento Editado, artículo en [arxiv](https://arxiv.org/abs/2406.02882).

Zengkui Sun, Yijin Liu, Jiaan Wang, Fandong Meng, Jinan Xu, Yufeng Chen$^{\dagger}$, Jie Zhou

Nuestro código sigue a [EasyEdit](https://github.com/zjunlp/EasyEdit).

## Requisitos

Información detallada en requirements.txt, y los paquetes se pueden instalar de la siguiente manera:

```
pip install -r requirements.txt
```

## Datos

Los datos se pueden encontrar en [EasyEdit](https://github.com/zjunlp/EasyEdit), y los datos preprocesados directamente por nosotros se pueden encontrar en [Google Drive](https://drive.google.com/file/d/1hHDXjCi78r3ksF87-ecJKHHmk6mvJi3s/view?usp=drive_link) (descomprimir en el directorio `data`).

## Modelos

Utilizamos `gpt-j-6b`, `llama-2-7b` y `llama-2-13b` como los LLM en nuestro artículo.

Tenga en cuenta que puede modificar la ruta del modelo preentrenado estableciendo el valor de `plm_dir` en nuestros scripts.

## Entrenamiento / Inferencia

Mostramos los scripts en `scripts`.

Por ejemplo, puede ejecutar DISCO (nuestro método) con `gpt-j-6b` de la siguiente manera:

```
bash scripts/zsre_disco.sh
```

Una vez finalizada la evaluación de todo el conjunto, los resultados se guardarán en `$ckpt_dir/`, ruta que puede modificarse en los scripts.

## Evaluación

Evaluamos `F1 / EM` con `eval/eval.sh`, `OE / TE` con `eval/port_eval.sh`, y el ejemplo completo se puede encontrar en `eval/gen_metrics.sh`.

## BibTex

Si encuentra este repositorio útil para su investigación, considere citar nuestro artículo:

```
@article{sun2024outdated,
  title={Outdated Issue Aware Decoding for Factual Knowledge Editing},
  author={Sun, Zengkui and Liu, Yijin and Wang, Jiaan and Meng, Fandong and Xu, Jinan and Chen, Yufeng and Zhou, Jie},
  journal={arXiv preprint arXiv:2406.02882},
  year={2024}
}
```
