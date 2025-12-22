"""Celery application instance for background task execution."""

from __future__ import annotations

from typing import TYPE_CHECKING

from celery import Celery

if TYPE_CHECKING:
    from ia_agent_fwk.config.settings import AppSettings


def get_celery_app(settings: AppSettings | None = None) -> Celery:
    """Create and configure a Celery application from *settings*.

    Parameters
    ----------
    settings:
        Application settings. When ``None``, settings are loaded via
        ``load_config()``.

    Returns
    -------
    Celery
        A configured Celery application instance.

    """
    if settings is None:
        from ia_agent_fwk.config.loader import load_config  # noqa: PLC0415

        settings = load_config()

    celery_settings = settings.execution.celery

    app = Celery(
        "ia_agent_fwk",
        broker=celery_settings.broker_url,
        backend=celery_settings.result_backend,
    )

    soft_limit = max(celery_settings.task_timeout - 30, 10)

    app.conf.update(
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],
        task_time_limit=celery_settings.task_timeout,
        task_soft_time_limit=soft_limit,
        result_expires=celery_settings.result_expires,
        worker_prefetch_multiplier=celery_settings.worker_prefetch_multiplier,
        worker_concurrency=celery_settings.worker_concurrency,
        task_acks_late=True,
        worker_hijack_root_logger=False,
    )

    # Auto-discover tasks from the execution.tasks module
    app.autodiscover_tasks(["ia_agent_fwk.execution"])

    return app


# Module-level instance (lazy -- created on first access by workers)
celery_app: Celery = get_celery_app()
