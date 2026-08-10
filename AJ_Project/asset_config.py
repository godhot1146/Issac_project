"""
asset_config.py — 에셋 폴더(isaac_assets) 경로를 컴퓨터마다 알아서 찾아준다.

우선순위:
  1) 환경변수 ISAAC_ASSETS 가 있으면 그걸 사용
  2) 이전에 저장해둔 로컬 설정 파일(config.local.json)  ← git에 커밋 안 함
  3) 흔한 위치 자동 탐색 (레포 나란히 배치 / 홈·Desktop 등)
  4) 그래도 못 찾으면 첫 실행 때 한 번만 물어보고 저장

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


def _candidates():
    """자동 탐색 후보들 (레포를 나란히 둔 경우 + 흔한 위치)."""
    home = os.path.expanduser("~")
    return [
        # 레포를 나란히 둔 경우: .../Issac_project/AJ_Project → .../Issac_asset/isaac_assets
        os.path.abspath(os.path.join(_HERE, "..", "..", "Issac_asset", "isaac_assets")),
        os.path.join(home, "Desktop", "Issac_asset", "isaac_assets"),
        os.path.join(home, "Issac_asset", "isaac_assets"),
    ]


def get_asset_root():
    # 1) 환경변수 우선
    env = os.environ.get("ISAAC_ASSETS")
    if _is_valid(env):
        return env

    # 2) 저장된 로컬 설정
    if os.path.exists(CONFIG_PATH):
        try:
            saved = json.load(open(CONFIG_PATH)).get("asset_root")
            if _is_valid(saved):
                return saved
        except (ValueError, OSError):
            pass  # 파일이 깨졌으면 무시하고 아래로

    # 3) 흔한 위치 자동 탐색 → 찾으면 저장
    for cand in _candidates():
        if _is_valid(cand):
            _save(cand)
            return cand

    # 4) 첫 실행: 한 번만 물어보고 저장
    print("[asset_config] 에셋 폴더(isaac_assets)를 찾지 못했습니다.")
    path = input("isaac_assets 폴더의 전체 경로를 입력하세요: ").strip()
    if not _is_valid(path):
        raise FileNotFoundError(
            f"유효한 에셋 폴더가 아닙니다(urdf 하위폴더가 없음): {path}"
        )
    _save(path)
    return path


if __name__ == "__main__":
    # 단독 실행 시 현재 해석되는 경로 확인용
    print("asset_root =", get_asset_root())
