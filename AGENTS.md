# Instrucciones del motor analítico

## Contexto

Este repositorio contiene los procesos batch Python de análisis y machine learning. La API Express y el frontend React viven en repositorios independientes. Antes de implementar una tarea, consultar el documento funcional correspondiente en `docs/`.

## Responsabilidad

- Implementar jobs reproducibles para extracción, features, entrenamiento, evaluación y scoring batch.
- No exponer una API HTTP ni entrar en el camino de una petición del usuario.
- Leer vistas o snapshots explícitamente acordados con el backend.
- Escribir resultados precalculados con versión, fecha y metadatos de evaluación.
- Mantener separadas extracción, transformación, entrenamiento, evaluación y persistencia.

## Integridad analítica

- Construir features point-in-time: ninguna fila puede usar información posterior al instante predicho.
- Separar train, validation y test por tiempo o usuario; documentar la elección.
- Comparar todo modelo contra un criterio de referencia simple.
- Preferir una regla cuando el modelo no aporte una mejora medible.
- Fijar seeds, versiones, parámetros y artefactos necesarios para reproducir resultados.
- No usar datos personales reales ni subir datasets sensibles, modelos grandes o notebooks con salidas privadas.
- Los notebooks son exploratorios; la lógica aceptada debe migrar a módulos y tests.
- Nombrar conceptos con los términos literales de `docs/D2-glosario.md`.

## Forma de trabajo

- Crear ramas desde `develop`; todo cambio entra por pull request.
- Usar Conventional Commits en inglés: `type(scope): summary`.
- Coordinar mediante PR relacionados cualquier cambio en la interfaz de datos con el backend.

## Verificación

- Ejecutar `python -m ruff check .`.
- Ejecutar `python -m mypy src`.
- Ejecutar `python -m pytest`.

## Code Review Rules

- Bloquear fuga temporal, métricas calculadas sobre el conjunto de entrenamiento o splits no reproducibles.
- Bloquear scores sin versión del componente y fecha de cálculo.
- Bloquear inferencia online o acoplamiento directo del frontend con Python.
- Exigir tests para transformaciones, límites temporales, valores faltantes y persistencia idempotente.
