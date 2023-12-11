from dataclasses import dataclass


@dataclass
class PFunAPIRoutes:
    FRONTEND_ROUTES = (
        '/run',
        '/run-at-time',
        '/params/schema',
        '/params/default'
    )

    PUBLIC_ROUTES = (
        '/',
        '/run',
        '/fit',
        '/run-at-time',
        '/routes',
        '/sdk',
        '/params/schema',
        '/params/default',
    )

    PRIVATE_ROUTES = (
        '/run',
        '/fit',
        '/run-at-time',
        '/sdk'
    )