"""
asset_config.py — 에셋 폴더(isaac_assets) 경로를 컴퓨터마다 알아서 찾아준다.

우선순위:
  1) 로컬 설정 파일(config.local.json)이 있으면 그 안의 경로를 사용  ← git에 커밋 안 함
  2) 없으면 환경변수 ISAAC_ASSETS 확인
       - 있으면 그 값을 config.local.json에 그대로 저장 후 사용
       - 없으면 config.local.json을 빈 값으로 생성하고, 경로를 채워넣으라는 안내 메시지 출력

런파일에서:
    from asset_config import get_asset_root
    asset_root = get_asset_root()
"""
import os
import json

# 이 파일 기준 경로들
_HERE = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(_HERE, "config.local.json")

# 유효한 isaac_assets 폴더인지 판별(대표 하위폴더로 확인)
def _is_valid(path):
    return bool(path) and os.path.isdir(os.path.join(path, "urdf"))


def _save(path):
    with open(CONFIG_PATH, "w") as f:
        json.dump({"asset_root": path}, f, indent=2)
    print(f"[asset_config] 에셋 경로 저장됨 → {CONFIG_PATH}")


def get_asset_root():
    # 1) config.local.json 부터 확인
    if os.path.exists(CONFIG_PATH):
        try:
            saved = json.load(open(CONFIG_PATH)).get("asset_root")
        except (ValueError, OSError):
            saved = None

        if _is_valid(saved):
            return saved

        raise FileNotFoundError(
            f"{CONFIG_PATH} 의 asset_root 경로가 유효하지 않습니다: {saved!r}\n"
            f"해당 파일을 열어 isaac_assets 폴더의 전체 경로를 입력한 뒤 다시 실행해주세요."
        )

    # 2) config.local.json이 없으면 환경변수 확인
    env = os.environ.get("ISAAC_ASSETS")
    if _is_valid(env):
        _save(env)
        return env

    # 환경변수도 없으면 빈 설정 파일만 생성하고 안내
    _save("")
    print(f"[asset_config] {CONFIG_PATH} 파일을 생성했습니다.")
    print("해당 파일의 'asset_root' 값에 isaac_assets 폴더의 전체 경로를 입력한 뒤 다시 실행해주세요.")
    raise FileNotFoundError(f"{CONFIG_PATH} 에 asset_root 경로를 입력해주세요.")


if __name__ == "__main__":
    # 단독 실행 시 현재 해석되는 경로 확인용
    print("asset_root =", get_asset_root())

