# CLAUDE.md

Leé primero `AGENTS.md` y el documento funcional correspondiente en `docs/`.

Este repositorio contiene únicamente procesos batch Python. No expongas HTTP ni ejecutes inferencia dentro de una petición del backend. Toda feature debe ser point-in-time y toda salida debe registrar versión, instante, contexto y métricas contra un criterio de referencia reproducible.

Los contratos de datos se coordinan con el backend mediante documentación y PR relacionados; nunca mediante imports entre repositorios.
