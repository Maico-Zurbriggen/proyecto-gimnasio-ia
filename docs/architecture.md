# Arquitectura del sistema

## Distribución

```text
React SPA ──REST/JSON──> Express API ──> PostgreSQL
                                             ^
                                             |
                                      Python batch
```

Este repositorio contiene únicamente el motor batch Python. La separación física no lo convierte en un servicio online.

## Fronteras

- No expone HTTP.
- Lee vistas o snapshots versionados acordados con el backend.
- Escribe resultados precalculados con versión, instante y explicación.
- No modifica fuentes transaccionales ni comparte código con el backend.
- Los cambios de interfaz de datos requieren documentación y PR relacionados.

## Pipeline

1. Extraer un snapshot con instante de corte explícito.
2. Validar esquema, calidad y ausencia de información futura.
3. Construir features point-in-time.
4. Entrenar y evaluar contra un criterio de referencia simple.
5. Persistir métricas y resultados versionados de forma idempotente.

## Invariantes

- Ninguna feature usa información posterior al instante predicho.
- Ninguna salida se publica sin versión y contexto reproducible.
- Falta de datos no equivale a cero.
- Los datos simulados se identifican y no se presentan como reales.
- Si un modelo no supera al criterio simple, se conserva la regla simple.
