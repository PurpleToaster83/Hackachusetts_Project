# Project Structure

## Current layout

```
Hackachusetts_Project/
├── .git/            # version control
├── .kiro/
│   └── steering/    # AI assistant steering docs (product.md, tech.md, structure.md)
└── README.md        # project overview
```

The project has no source directories yet.

## Conventions going forward

- Keep source code in a dedicated top-level directory (e.g. `src/`) rather than scattering it in the root.
- Group files by feature or responsibility; keep modules small and focused.
- Place tests alongside the code they cover or in a parallel `tests/` tree, following the convention of the chosen stack.
- Store configuration and secrets outside version control (use `.env` + `.gitignore`); never commit credentials.
- Update this file as the structure grows so it reflects the real layout.
