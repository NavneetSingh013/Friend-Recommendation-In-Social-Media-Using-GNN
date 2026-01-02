"""
Check if two users are already friends.
"""

import torch

def check_if_friends(user1_id, user2_id):
    """Check if two users are already friends."""
    data = torch.load('data/processed/facebook_combined.pt', weights_only=False)
    edge_index = data.edge_index
    
    # Check if edge exists
    user1_neighbors = set()
    user2_neighbors = set()
    
    for i in range(edge_index.size(1)):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        
        if src == user1_id:
            user1_neighbors.add(dst)
        if dst == user1_id:
            user1_neighbors.add(src)
        if src == user2_id:
            user2_neighbors.add(dst)
        if dst == user2_id:
            user2_neighbors.add(src)
    
    are_friends = (user2_id in user1_neighbors) or (user1_id in user2_neighbors)
    
    print("=" * 60)
    print(f"CHECKING IF USERS {user1_id} AND {user2_id} ARE FRIENDS")
    print("=" * 60)
    print(f"\nUser {user1_id} has {len(user1_neighbors)} friends")
    print(f"User {user2_id} has {len(user2_neighbors)} friends")
    print(f"\nAre they already friends? {are_friends}")
    
    if are_friends:
        print(f"\n[YES] Users {user1_id} and {user2_id} are ALREADY FRIENDS")
        print("This explains why:")
        print(f"  - User {user1_id} CAN recommend User {user2_id} (they're not friends from 1043's perspective)")
        print(f"  - User {user2_id} CANNOT recommend User {user1_id} (1043 is excluded from 1254's candidate set)")
    else:
        print(f"\n[NO] Users {user1_id} and {user2_id} are NOT friends")
        print("The asymmetric recommendation is due to:")
        print("  - Different prediction scores in each direction")
        print("  - Different candidate pools or ranking")
        print("  - Asymmetric graph structure/neighborhoods")
    
    print("=" * 60)

if __name__ == '__main__':
    check_if_friends(1043, 1254)

