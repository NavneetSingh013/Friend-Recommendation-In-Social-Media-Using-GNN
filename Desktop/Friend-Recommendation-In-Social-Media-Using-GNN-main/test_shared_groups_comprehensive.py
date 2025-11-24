"""
Comprehensive test script for shared groups functionality.
Tests the shared groups computation with various user pairs.
"""

import torch
import sys
import os
sys.path.append('.')

from src.evaluation.explainability import compute_shared_groups, explain_recommendation

def test_shared_groups():
    """Test shared groups computation."""
    print("=" * 70)
    print("SHARED GROUPS TEST SUITE")
    print("=" * 70)
    
    # Load Facebook dataset
    print("\n1. Loading Facebook dataset...")
    try:
        data = torch.load('data/processed/facebook_combined.pt', weights_only=False)
        print(f"   [OK] Loaded dataset: {data.num_nodes} nodes, {data.edge_index.size(1) // 2} edges")
        print(f"   [OK] Has circles: {hasattr(data, 'circles') and bool(data.circles)}")
        if hasattr(data, 'circles') and data.circles:
            print(f"   [OK] Number of circles/groups: {len(data.circles)}")
    except FileNotFoundError:
        print("   [ERROR] Facebook dataset not found!")
        print("   Please run: python scripts/download_and_prepare.py --dataset facebook --download --preprocess")
        return
    
    # Test 1: Find users with shared groups
    print("\n2. Finding users with shared groups...")
    circle_members = {}
    for circle_name, members in data.circles.items():
        for member in members:
            if member not in circle_members:
                circle_members[member] = []
            circle_members[member].append(circle_name)
    
    # Find pairs with shared groups
    shared_pairs = []
    member_list = list(circle_members.keys())
    for i, u1 in enumerate(member_list[:50]):  # Test first 50 users
        for u2 in member_list[i+1:min(i+51, len(member_list))]:
            shared = set(circle_members[u1]) & set(circle_members[u2])
            if shared:
                # Convert to mapped indices
                if hasattr(data, 'node_to_idx'):
                    idx1 = data.node_to_idx.get(u1, -1)
                    idx2 = data.node_to_idx.get(u2, -1)
                    if idx1 >= 0 and idx2 >= 0:
                        shared_pairs.append((idx1, idx2, u1, u2, list(shared)))
    
    print(f"   [OK] Found {len(shared_pairs)} pairs with shared groups")
    
    if not shared_pairs:
        print("   [ERROR] No pairs with shared groups found!")
        return
    
    # Test 2: Test shared groups computation
    print("\n3. Testing shared groups computation...")
    test_cases = shared_pairs[:10]  # Test first 10 pairs
    passed = 0
    failed = 0
    
    for idx1, idx2, orig1, orig2, expected_groups in test_cases:
        computed_groups = compute_shared_groups(data, idx1, idx2)
        expected_count = len(expected_groups)
        computed_count = len(computed_groups)
        
        if computed_count == expected_count:
            print(f"   [PASS] Users {idx1} & {idx2} (original IDs {orig1} & {orig2}): {computed_count} shared groups")
            passed += 1
        else:
            print(f"   [FAIL] Users {idx1} & {idx2}: Expected {expected_count}, got {computed_count}")
            print(f"     Expected: {expected_groups}")
            print(f"     Got: {computed_groups}")
            failed += 1
    
    print(f"\n   Results: {passed} passed, {failed} failed")
    
    # Test 3: Test with explain_recommendation
    print("\n4. Testing explain_recommendation function...")
    if test_cases:
        idx1, idx2, orig1, orig2, expected_groups = test_cases[0]
        explanation = explain_recommendation(data, idx1, idx2, 0.95)
        
        print(f"   Testing with users {idx1} & {idx2}:")
        print(f"   - Shared groups: {explanation['num_shared_groups']}")
        print(f"   - Group names: {explanation['shared_groups']}")
        print(f"   - Mutual friends: {explanation['mutual_friends']}")
        print(f"   - Profile similarity: {explanation['profile_similarity']:.4f}")
        print(f"   - Path evidence: {explanation['path_evidence']}")
        
        if explanation['num_shared_groups'] > 0:
            print("   [OK] explain_recommendation correctly shows shared groups")
        else:
            print("   [WARNING] explain_recommendation shows 0 shared groups")
    
    # Test 4: Test edge cases
    print("\n5. Testing edge cases...")
    
    # Test with users not in any circles
    test_user1 = 0
    test_user2 = 1
    groups = compute_shared_groups(data, test_user1, test_user2)
    print(f"   Users {test_user1} & {test_user2} (may not be in circles): {len(groups)} shared groups")
    
    # Test with same user
    if hasattr(data, 'circles') and data.circles:
        # Find a user in a circle
        first_circle = list(data.circles.values())[0]
        if first_circle:
            orig_id = first_circle[0]
            if hasattr(data, 'node_to_idx'):
                user_idx = data.node_to_idx.get(orig_id, -1)
                if user_idx >= 0:
                    groups = compute_shared_groups(data, user_idx, user_idx)
                    print(f"   Same user {user_idx}: {len(groups)} shared groups (should be all their groups)")
    
    # Test 5: Summary statistics
    print("\n6. Summary Statistics...")
    if shared_pairs:
        group_counts = [len(groups) for _, _, _, _, groups in shared_pairs]
        print(f"   - Pairs with shared groups: {len(shared_pairs)}")
        print(f"   - Average shared groups per pair: {sum(group_counts) / len(group_counts):.2f}")
        print(f"   - Max shared groups: {max(group_counts)}")
        print(f"   - Min shared groups: {min(group_counts)}")
        
        # Count pairs by number of shared groups
        from collections import Counter
        count_dist = Counter(group_counts)
        print(f"   - Distribution:")
        for count, freq in sorted(count_dist.items()):
            print(f"     {count} shared groups: {freq} pairs")
    
    print("\n" + "=" * 70)
    print("TEST COMPLETE")
    print("=" * 70)
    
    # Test cases for manual verification
    print("\n7. Test Cases for Manual Verification:")
    print("   Use these user pairs in the Streamlit app to verify shared groups:")
    print()
    for i, (idx1, idx2, orig1, orig2, groups) in enumerate(shared_pairs[:5], 1):
        print(f"   Test Case {i}:")
        print(f"   - User 1 ID: {idx1} (original: {orig1})")
        print(f"   - User 2 ID: {idx2} (original: {orig2})")
        print(f"   - Expected shared groups: {len(groups)}")
        print(f"   - Group names: {groups}")
        print()

if __name__ == '__main__':
    test_shared_groups()

