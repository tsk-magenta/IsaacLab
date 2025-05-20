#!/usr/bin/env python3
"""
HDF5 파일 구조 확인 스크립트
annotated_test.hdf5 파일의 전체 구조를 출력합니다.
"""
# /home/hys/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/paint/usd_folder/explore_hdf5.py
# python explore_hdf5.py --file /home/hys/IsaacLab/datasets/annotated_dataset.hdf5 
import h5py
import numpy as np
import argparse
import os
from typing import Optional, Any, Dict, List, Tuple


def print_attrs(name: str, obj: Any) -> None:
    """객체의 모든 속성을 출력합니다."""
    print(f"\nAttributes of '{name}':")
    for key, val in obj.attrs.items():
        print(f"  {key}: {val}")


def explore_hdf5(file_path: str, max_array_items: int = 5, 
                show_attributes: bool = True, 
                show_dataset_shape: bool = True, 
                show_dataset_values: bool = True) -> None:
    """
    HDF5 파일 구조를 탐색하고 출력합니다.
    
    Args:
        file_path: HDF5 파일 경로
        max_array_items: 배열 데이터셋에서 출력할 최대 항목 수
        show_attributes: 속성을 출력할지 여부
        show_dataset_shape: 데이터셋 형태를 출력할지 여부
        show_dataset_values: 데이터셋 값을 출력할지 여부
    """
    def visitor_func(name: str, node: Any) -> None:
        indent = '  ' * name.count('/')
        
        if isinstance(node, h5py.Group):
            print(f"{indent}Group: {name}")
            if show_attributes and len(node.attrs) > 0:
                print(f"{indent}  Attributes:")
                for key, val in node.attrs.items():
                    print(f"{indent}    {key}: {val}")
        
        elif isinstance(node, h5py.Dataset):
            shape_str = str(node.shape) if show_dataset_shape else ""
            dtype_str = str(node.dtype)
            print(f"{indent}Dataset: {name} {shape_str} {dtype_str}")
            
            if show_attributes and len(node.attrs) > 0:
                print(f"{indent}  Attributes:")
                for key, val in node.attrs.items():
                    print(f"{indent}    {key}: {val}")
            
            if show_dataset_values:
                if len(node.shape) == 0:  # 스칼라 데이터셋
                    print(f"{indent}  Value: {node[()]}")
                else:
                    # 첫 몇 개 항목만 출력
                    if node.size > 0:
                        # 배열의 첫 부분 가져오기
                        try:
                            if len(node.shape) == 1:
                                # 1차원 배열일 경우
                                if node.dtype.kind in ('i', 'f', 'u'):  # 숫자 타입
                                    sample = node[:min(max_array_items, len(node))]
                                    print(f"{indent}  First {len(sample)} values: {sample}")
                                elif node.dtype.kind == 'S' or node.dtype.kind == 'U':  # 문자열 타입
                                    sample = [item.decode('utf-8') if isinstance(item, bytes) else item 
                                             for item in node[:min(max_array_items, len(node))]]
                                    print(f"{indent}  First {len(sample)} values: {sample}")
                                else:
                                    print(f"{indent}  Data type '{node.dtype}' preview not supported")
                            else:
                                # 다차원 배열일 경우
                                print(f"{indent}  Multi-dimensional array (shape: {node.shape})")
                        except Exception as e:
                            print(f"{indent}  Error previewing data: {e}")

    try:
        with h5py.File(file_path, 'r') as f:
            # 파일 자체의 속성
            if show_attributes and len(f.attrs) > 0:
                print("File attributes:")
                for key, val in f.attrs.items():
                    print(f"  {key}: {val}")
                print("\n")
                
            # 모든 항목 방문
            f.visititems(visitor_func)
            
            # 루트 그룹 자체도 방문
            print("\nRoot group:")
            for key, val in f.items():
                print(f"  {key}: {'Group' if isinstance(val, h5py.Group) else 'Dataset'}")
                
    except Exception as e:
        print(f"Error exploring HDF5 file: {e}")


def summarize_episodes(file_path: str) -> None:
    """
    HDF5 파일에서 에피소드 정보를 요약합니다.
    
    Args:
        file_path: HDF5 파일 경로
    """
    try:
        with h5py.File(file_path, 'r') as f:
            # 환경 이름 및 버전 확인
            env_name = f.attrs.get('env_name', 'Not specified')
            print(f"Environment: {env_name}")
            
            # 에피소드 그룹 찾기
            episode_count = 0
            episodes_group = None
            
            if 'episodes' in f:
                episodes_group = f['episodes']
                episode_count = len(episodes_group)
            
            print(f"Total episodes: {episode_count}")
            
            if episodes_group is not None and episode_count > 0:
                print("\nEpisode summary:")
                
                for episode_name, episode in episodes_group.items():
                    print(f"\n  Episode: {episode_name}")
                    
                    # 성공 여부
                    if 'success' in episode.attrs:
                        print(f"    Success: {episode.attrs['success']}")
                    
                    # 액션 및 상태 데이터 형태
                    if 'data' in episode:
                        data_group = episode['data']
                        print("    Data:")
                        
                        if 'actions' in data_group:
                            actions = data_group['actions']
                            print(f"      Actions: shape={actions.shape}, dtype={actions.dtype}")
                        
                        if 'initial_state' in data_group:
                            initial_state = data_group['initial_state']
                            print(f"      Initial state: shape={initial_state.shape}, dtype={initial_state.dtype}")
                        
                        # 서브태스크 정보 확인
                        if 'obs' in data_group:
                            obs_group = data_group['obs']
                            if 'datagen_info' in obs_group:
                                datagen_info = obs_group['datagen_info']
                                if 'subtask_term_signals' in datagen_info:
                                    subtask_signals = datagen_info['subtask_term_signals']
                                    print("      Subtask signals:")
                                    for signal_name, signal_data in subtask_signals.items():
                                        # 신호가 True인 타임스텝 찾기
                                        true_indices = np.where(signal_data[:])[0]
                                        if len(true_indices) > 0:
                                            print(f"        {signal_name}: Completed at steps {true_indices}")
                                        else:
                                            print(f"        {signal_name}: Not completed")
                    
                    # 에피소드 길이
                    if 'data' in episode and 'actions' in episode['data']:
                        actions = episode['data']['actions']
                        print(f"    Episode length: {len(actions)} steps")
    
    except Exception as e:
        print(f"Error summarizing episodes: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explore HDF5 file structure")
    parser.add_argument("--file", default="./datasets/dataset_annotated.hdf5", 
                       help="Path to the HDF5 file")
    parser.add_argument("--max_items", type=int, default=5, 
                       help="Maximum number of array items to display")
    parser.add_argument("--no_attrs", action="store_true", 
                       help="Do not show attributes")
    parser.add_argument("--no_shape", action="store_true", 
                       help="Do not show dataset shapes")
    parser.add_argument("--no_values", action="store_true", 
                       help="Do not show dataset values")
    parser.add_argument("--summary", action="store_true", 
                       help="Show episode summary instead of full structure")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file):
        print(f"Error: File '{args.file}' does not exist.")
        exit(1)
    
    print(f"Exploring HDF5 file: {args.file}\n")
    
    if args.summary:
        summarize_episodes(args.file)
    else:
        explore_hdf5(
            args.file, 
            max_array_items=args.max_items,
            show_attributes=not args.no_attrs,
            show_dataset_shape=not args.no_shape,
            show_dataset_values=not args.no_values
        )
    
    print("\nExploration complete.")