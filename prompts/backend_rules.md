### Backend-specific (FastAPI, Express, Django, Flask):
- **Async I/O**: Use async endpoints for database/HTTP/file operations. Blocking I/O in async handlers kills throughput.
- **Input validation**: Pydantic models (FastAPI), zod/joi (Express), serializers (Django) — never manual string parsing.
- **HTTP status codes**: 201 for create, 204 for delete, 404 for not found, 422 for validation error. Not everything is 200.
- **CORS/security**: Add CORS middleware when frontend is separate. Add security headers (helmet, django-cors-headers).
- **Database**: Use ORM migrations — never raw CREATE TABLE in production code. Pin dependency versions.
