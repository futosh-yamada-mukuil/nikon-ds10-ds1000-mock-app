"""Keep source and packaged applications on the same entry point."""

from app.__main__ import main


if __name__ == "__main__":
    raise SystemExit(main())
