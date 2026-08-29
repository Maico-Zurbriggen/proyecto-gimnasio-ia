# Instrucciones del servicio de IA y analítica

## Contexto

<<<<<<< Updated upstream
Este repositorio contiene los procesos batch Python de análisis y machine learning. La API Express y el frontend React viven en repositorios independientes. Antes de implementar una tarea, consultar el documento funcional correspondiente en `docs/`.
=======
Este repositorio contiene dos límites: servicio generativo online Python y jobs analíticos batch. La API y el worker se despliegan en el Polo, se publican al backend mediante ngrok y orquestan el LLM alojado allí. Frontend nunca consume este repositorio directamente.

La documentación canónica vive en `Maico-Zurbriggen/proyecto-gimnasio-documentacion`. Con repositorios hermanos, leer primero `../proyecto-gimnasio-documentacion/AGENTS.md` y usar `manifest.json`. Si no está local, consultar GitHub; no copiar documentación aquí.
>>>>>>> Stashed changes

## Servicio generativo

- Mantener separados API, autenticación, contratos, orquestación, conector LLM, persistencia y worker.
- Publicar OpenAPI versionado como fuente de verdad para backend.
- Aceptar solicitudes idempotentes con `202`; nunca esperar al LLM dentro de la petición.
- Procesar mediante worker durable capaz de recuperar trabajos tras un reinicio.
- Llamar al LLM sólo mediante un conector privado y validar su salida estructural.
- Escribir únicamente estados y resultados en estructuras de integración acordadas.
- No crear, aprobar, asignar ni activar rutinas; backend conserva reglas y autoridad.
- Cada intento vence inicialmente a los 120 segundos y admite un único reintento.
- Tras el segundo fallo registrar indisponibilidad; no generar fallback determinístico.

## Datos y seguridad

<<<<<<< Updated upstream
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
=======
- Usar identificadores técnicos y sólo contexto necesario.
- No recibir ni registrar nombre, correo, teléfono, documento, credenciales o prompts completos.
- Separar authtoken de ngrok, credencial backend–IA y credencial IA–LLM.
- Credenciales test/producción seleccionan conexiones configuradas internamente; nunca aceptar una URL de base en la petición.
- El rol PostgreSQL de IA no accede a identidad ni modifica tablas de dominio.
- Resultados fallidos y solicitudes abandonadas se eliminan a los 30 días.

## Analítica batch

- Mantener separados extracción, features, entrenamiento, evaluación, scoring y persistencia.
- Construir features point-in-time y splits reproducibles.
- Comparar modelos contra un criterio simple.
- Escribir resultados precalculados con versión, fecha y metadatos.
- No ejecutar jobs batch dentro de una petición online.

## Desarrollo y pruebas

- El desarrollo local usa Neon Test con rol restringido y, cuando sea accesible, la API real del LLM del Polo.
- No implementar un adaptador fake ejecutable. En tests se permiten dobles del conector LLM y de persistencia.
- No ejecutar migraciones; backend es dueño del esquema.
- Versionar modelo, prompt, parámetros y contrato.
- Todo cambio de modelo, prompt o parámetros requiere dataset de regresión verde y validación de al menos un entrenador antes de `main`.

## Forma de trabajo

- Crear ramas desde `develop`; todo cambio entra por PR.
- Promover `develop → test → main`; no crear commits exclusivos en `test`.
- Usar Conventional Commits en inglés.
- Coordinar PR relacionados con backend y documentación al cambiar OpenAPI, persistencia, métricas o flujos.
>>>>>>> Stashed changes

## Verificación

- Ejecutar `python -m ruff check .`.
- Ejecutar `python -m mypy src`.
- Ejecutar `python -m pytest`.
- Probar idempotencia, reinicio del worker, timeout, reintento, permisos PostgreSQL y aislamiento de ambientes.

## Code Review Rules

- Bloquear espera del LLM dentro de la petición HTTP.
- Bloquear acceso a tablas de dominio o datos identificatorios innecesarios.
- Bloquear resultados sin modelo, configuración, instante y contexto reproducible.
- Bloquear cambios de IA sin evaluación y revisión humana requerida.
- Bloquear fuga temporal o splits no reproducibles en jobs batch.
