# Rag Sample

- Local 환경에서 `RAG`를 구현하기 위한 샘플 리포지토리입니다.

## 환경 설정

- `Python 3.7`, `Rust 1.6` 이상의 버전이 설치되어 있어야 합니다. (https://pyo3.rs/v0.22.5/#usage)
  - `Python` 버전 확인
    ```bash
    $ python --version
    ```
  - `Rust` 버전 확인
    ```bash
    $ rustc --version
    ```
- `Tesseract` 설치
  - 참고: https://kongda.tistory.com/93
  - `Windows`
    - [여기](https://digi.bib.uni-mannheim.de/tesseract/) 최하단 최신 버전을 다운로드 한 후 설치
    - 설치 후 설치 경로에 환경 변수 설정 후 `cmd` 창을 열어 `tesseract --version` 명령어를 실행하여 설치 여부 확인
  - `macOS`
    - `Homebrew`를 이용하여 설치
      ```bash
      $ brew install tesseract
      ```
    - 설치 후 `tesseract --version` 명령어를 실행하여 설치 여부 확인
    - `macOS`에서는 `/opt/homebrew/Cellar/tesseract/{version}/share/tessdata` 경로에 언어 `trainning data`를 설치할 수 있습니다.

## 문제 해결