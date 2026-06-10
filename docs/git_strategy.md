# Estrategia de Git: GitHub Flow

## Flujo de trabajo utilizado

Este proyecto sigue la metodología **GitHub Flow**, una estrategia simple y efectiva para proyectos de ciencia de datos.

## Ramas principales

| Rama | Propósito |
|------|-----------|
| `main` | Código estable y listo para producción. Solo recibe merges desde `development`. |
| `development` | Rama de desarrollo activo. Aquí se construyen features antes de fusionarlos. |

## Convención de commits

Se usa el estándar **Conventional Commits**:

```
<tipo>(<alcance>): <descripción corta>
```

| Tipo | Cuándo usar |
|------|-------------|
| `feat` | Nueva funcionalidad |
| `fix` | Corrección de bug |
| `docs` | Cambios en documentación |
| `refactor` | Reestructuración de código |
| `data` | Cambios en datos o artefactos |
| `chore` | Tareas de mantenimiento |

**Ejemplos:**
```
feat(notebooks): add preprocessing notebook with EDA
fix(model): correct label encoding for categorical features
docs(readme): add model card section
data(artifacts): add trained random forest model pickle
```

## Flujo de trabajo paso a paso

```
main
 └── development
       └── [trabajo directo en development para este proyecto]
             └── Pull Request → main (al completar una versión)
```

1. Todo el desarrollo ocurre en la rama `development`
2. Al completar un hito importante se abre un **Pull Request** de `development` → `main`
3. El PR describe los cambios, el modelo entrenado y las métricas obtenidas
4. Una vez aprobado, se hace **merge** y se crea un **Release**

## Release v1.0.0

La versión 1.0.0 corresponde a la entrega final del Proyecto Curso I, incluyendo:
- Preprocesamiento completo de datos
- Modelo de ML entrenado y evaluado
- Integración con LLM para explicación de predicciones
- README con documentación completa
