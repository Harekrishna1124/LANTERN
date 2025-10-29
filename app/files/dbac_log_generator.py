# import random
# from datetime import datetime, timedelta
# import collections

# # --- Configuration ---
# TARGET_LOG_ENTRIES = 1000
# USER_FILE = "dac_user.txt"
# OBJECT_FILE = "dac_object.txt"
# LOG_FILE = "dac_logs.txt"

# # --- 1. Define User and Object Attributes ---
# # Simplified for a clear DAC example
# DEPARTMENTS = {
#     "Finance": ["Analyst", "Manager"],
#     "Sales": ["Representative", "Manager"],
#     "IT": ["Support_Specialist", "System_Admin"],
# }
# OBJECTS_STRUCTURE = {
#     "Financial_Report": ["Internal", "High"],
#     "Sales_Lead_List": ["Internal", "Medium"],
#     "System_Log": ["Confidential", "High"],
# }
# ACTIONS = ["read", "write", "delete"]

# # This dictionary maps an object type to the department that "owns" it.
# OWNING_DEPARTMENT = {
#     "Financial_Report": "Finance",
#     "Sales_Lead_List": "Sales",
#     "System_Log": "IT",
# }

# ACCESS_CONTROL_LISTS = {}

# # --- 2. The DAC "Ground Truth" Decision Logic ---
# def get_decision(user, obj, action):
#     user_permissions = ACCESS_CONTROL_LISTS.get(obj['object_name'], {}).get(user['user_name'], set())
#     return "Allow" if action in user_permissions else "Deny"

# # --- 3. Data Generation Functions (with Realistic Ownership) ---
# def generate_users():
#     users = []
#     for dept, designations in DEPARTMENTS.items():
#         for i in range(5):
#             user_name = f"{dept.lower()}_{designations[i % len(designations)].lower()}_{i}"
#             users.append({"user_name": user_name, "department": dept, "designation": designations[i % len(designations)]})
#     return users

# def generate_objects_and_acls(all_users):
#     print("Generating objects and realistic, department-based ACLs...")
#     objects = []
    
#     for o_type, sensitivities in OBJECTS_STRUCTURE.items():
#         for i in range(10):
#             object_name = f"{o_type.lower()}_{i}.txt"
#             objects.append({"object_name": object_name, "type": o_type, "sensitivity": random.choice(sensitivities)})
            
#             # --- CRITICAL FIX: Realistic ACL Generation Logic ---
#             relevant_dept = OWNING_DEPARTMENT[o_type]
            
#             # 1. Choose owner ONLY from the relevant department.
#             possible_owners = [u for u in all_users if u['department'] == relevant_dept]
#             if not possible_owners: continue
#             owner = random.choice(possible_owners)
#             acl = {owner['user_name']: {'read', 'write', 'delete'}} # Owner gets full control
            
#             # 2. Choose collaborators with a strong departmental bias.
#             num_collaborators = random.randint(1, 3)
#             for _ in range(num_collaborators):
#                 # 70% chance to pick a collaborator from the same department
#                 if random.random() < 0.7:
#                     collaborator_pool = [u for u in all_users if u['department'] == relevant_dept and u['user_name'] not in acl]
#                 # 30% chance to pick from any other department (cross-functional)
#                 else:
#                     collaborator_pool = [u for u in all_users if u['department'] != relevant_dept and u['user_name'] not in acl]
                
#                 if collaborator_pool:
#                     collaborator = random.choice(collaborator_pool)
#                     acl[collaborator['user_name']] = {'read'} # Collaborators get read-only
            
#             ACCESS_CONTROL_LISTS[object_name] = acl
            
#     return objects

# def format_for_file(entity, is_user=True):
#     if is_user: return f"<{entity['user_name']} department:{entity['department']} designation:{entity['designation']}>"
#     else: return f"<{entity['object_name']} type:{entity['type']} sensitivity:{entity['sensitivity']}>"

# # --- 4. Main Script Execution ---
# # --- 4. Main Script Execution (Corrected for precise counts and ratios) ---
# def main():
#     print("--- Starting DAC Log Generation with Realistic Correlations ---")

#     all_users = generate_users()
#     all_objects = generate_objects_and_acls(all_users)

#     with open(USER_FILE, "w") as f: f.write("\n".join([format_for_file(u) for u in all_users]))
#     with open(OBJECT_FILE, "w") as f: f.write("\n".join([format_for_file(o, is_user=False) for o in all_objects]))
    
#     generated_logs = []
#     start_time = datetime.now() - timedelta(hours=12)
    
#     # --- CRITICAL FIX: Use deterministic loops to guarantee counts ---
#     allow_ratio = 0.90
#     num_allow = int(TARGET_LOG_ENTRIES * allow_ratio)
#     num_deny = TARGET_LOG_ENTRIES - num_allow

#     # 1. Generate the exact number of "Allow" logs needed
#     print(f"Generating exactly {num_allow} targeted 'Allow' logs...")
#     allow_count = 0
#     while allow_count < num_allow:
#         target_obj_name = random.choice(list(ACCESS_CONTROL_LISTS.keys()))
#         acl_for_obj = ACCESS_CONTROL_LISTS.get(target_obj_name, {})
#         if not acl_for_obj: continue

#         user_with_permission_name = random.choice(list(acl_for_obj.keys()))
#         user = next((u for u in all_users if u['user_name'] == user_with_permission_name), None)
#         target_obj = next((o for o in all_objects if o['object_name'] == target_obj_name), None)
        
#         if not user or not target_obj: continue

#         action = random.choice(list(acl_for_obj[user_with_permission_name]))
        
#         time_str = (start_time + timedelta(seconds=allow_count * 20)).strftime("%H:%M:%S")
#         generated_logs.append(f"<{user['user_name']} {target_obj['object_name']} {action} {time_str} Allow>")
#         allow_count += 1

#     # 2. Generate the exact number of "Deny" logs needed
#     print(f"Generating exactly {num_deny} targeted 'Deny' logs...")
#     deny_count = 0
#     while deny_count < num_deny:
#         user, obj = random.choice(all_users), random.choice(all_objects)
#         allowed_actions = ACCESS_CONTROL_LISTS.get(obj['object_name'], {}).get(user['user_name'], set())
#         denied_actions = [a for a in ACTIONS if a not in allowed_actions]
        
#         # This loop will now continue until it finds a valid Deny case
#         if denied_actions:
#             action = random.choice(denied_actions)
#             time_str = (start_time + timedelta(seconds=(num_allow + deny_count) * 20)).strftime("%H:%M:%S")
#             generated_logs.append(f"<{user['user_name']} {obj['object_name']} {action} {time_str} Deny>")
#             deny_count += 1

#     random.shuffle(generated_logs)

#     print(f"\nWriting {len(generated_logs)} logs to {LOG_FILE}.")
#     with open(LOG_FILE, "w") as f: f.write("\n".join(generated_logs))
        
#     decision_counts = collections.Counter(log.split()[-1].replace("'", "").replace('>', '') for log in generated_logs)
#     total_count = sum(decision_counts.values()) or 1
    
#     print("\n--- Generation Complete for DAC dataset! ---")
#     print(f"Final Log Ratio: {decision_counts['Allow']} Allow ({decision_counts['Allow']/total_count:.1%}) | {decision_counts['Deny']} Deny ({decision_counts['Deny']/total_count:.1%})")

# if __name__ == "__main__":
#     main()
# import random
# from datetime import datetime, timedelta
# import collections

# # --- Configuration ---
# TARGET_LOG_ENTRIES = 5000
# USER_FILE = "dac_user.txt"
# OBJECT_FILE = "dac_object.txt"
# LOG_FILE = "dac_logs.txt"

# # --- 1. Define Data Structures (Mirrors the ABAC Generator for Fair Comparison) ---

# DEPARTMENTS = {
#     "Finance": ["Analyst", "Senior_Analyst", "Manager", "Director"],
#     "Sales": ["Representative", "Senior_Representative", "Manager", "Director"],
#     "IT": ["Support_Specialist", "System_Admin", "Network_Engineer", "IT_Manager"],
#     "HR": ["Generalist", "Recruiter", "HR_Manager", "Director"],
#     "Engineering": ["Software_Engineer", "Senior_Engineer", "Lead_Engineer", "Engineering_Manager"],
#     "Marketing": ["Coordinator", "Specialist", "Marketing_Manager", "Director"],
#     "Legal": ["Paralegal", "Analyst", "General_Counsel", "Chief_Legal_Officer"],
#     "Operations": ["Coordinator", "Operations_Manager", "Director_of_Operations"],
#     "Customer_Support": ["Agent", "Team_Lead", "Support_Manager"],
#     "Research": ["Researcher", "Senior_Researcher", "Lead_Scientist"]
# }

# OBJECTS_STRUCTURE = {
#     "Financial": ["Low", "Medium", "High"], "Operational": ["Low", "Medium"],
#     "System": ["Medium", "High"], "HR": ["Medium", "High", "Confidential"],
#     "Source_Code": ["Medium", "High"], "Legal": ["Medium", "High", "Confidential"],
#     "Marketing": ["Low", "Medium"], "Customer_Data": ["High", "Confidential"], "General": ["Low"]
# }

# ACTIONS = ["read", "write", "execute", "delete"]

# # This dictionary maps an object type to the department that logically "owns" it.
# OWNING_DEPARTMENT = {
#     "Financial": "Finance", "Operational": "Sales", "System": "IT",
#     "HR": "HR", "Source_Code": "Engineering", "Legal": "Legal",
#     "Marketing": "Marketing", "Customer_Data": "Customer_Support", "General": "HR",
# }

# ACCESS_CONTROL_LISTS = {}

# # --- 2. The DAC "Ground Truth" Decision Logic ---
# def get_decision(user, obj, action):
#     user_permissions = ACCESS_CONTROL_LISTS.get(obj['object_name'], {}).get(user['user_name'], set())
#     return "Allow" if action in user_permissions else "Deny"

# # --- 3. Data Generation Functions (with Realistic Ownership) ---
# def generate_users():
#     users = []
#     for dept, designations in DEPARTMENTS.items():
#         for i in range(len(designations) * 2):
#             user_name = f"{random.choice(['alex','casey','drew','jordan']).lower()}_{dept.lower()}_{i}"
#             users.append({"user_name": user_name, "department": dept, "designation": random.choice(designations)})
#     return users

# def generate_objects_and_acls(all_users):
#     print("Generating objects and realistic, department-based ACLs...")
#     objects = []
    
#     # --- CRITICAL FIX: This loop now matches the ABAC generator to produce 60 objects ---
#     for o_type, sensitivities in OBJECTS_STRUCTURE.items():
#         for sens in sensitivities:
#             for i in range(3):
#                 object_name = f"{o_type.lower()}_{sens.lower()}_{i}.dat"
#                 objects.append({"object_name": object_name, "type": o_type, "sensitivity": sens})
                
#                 relevant_dept = OWNING_DEPARTMENT.get(o_type, "IT")
                
#                 possible_owners = [u for u in all_users if u['department'] == relevant_dept]
#                 if not possible_owners: continue
#                 owner = random.choice(possible_owners)
#                 acl = {owner['user_name']: {'read', 'write', 'delete'}}
                
#                 num_collaborators = random.randint(3, 5)
#                 for _ in range(num_collaborators):
#                     if random.random() < 0.6:
#                         pool = [u for u in all_users if u['department'] == relevant_dept and u['user_name'] not in acl]
#                     else:
#                         pool = [u for u in all_users if u['department'] != relevant_dept and u['user_name'] not in acl]
                    
#                     if pool:
#                         collaborator = random.choice(pool)
#                         acl[collaborator['user_name']] = {'read'}
                
#                 ACCESS_CONTROL_LISTS[object_name] = acl
            
#     return objects

# def format_for_file(entity, is_user=True):
#     if is_user: return f"<{entity['user_name']} department:{entity['department']} designation:{entity['designation']}>"
#     else: return f"<{entity['object_name']} type:{entity['type']} sensitivity:{entity['sensitivity']}>"

# # --- 4. Main Script Execution (Corrected for precise counts and ratios) ---
# def main():
#     print("--- Starting DAC Log Generation (Fair Comparison) ---")

#     all_users = generate_users()
#     all_objects = generate_objects_and_acls(all_users)

#     with open(USER_FILE, "w") as f: f.write("\n".join([format_for_file(u) for u in all_users]))
#     with open(OBJECT_FILE, "w") as f: f.write("\n".join([format_for_file(o, is_user=False) for o in all_objects]))
    
#     generated_logs = []
#     start_time = datetime.now() - timedelta(hours=12)
    
#     allow_ratio = 0.90
#     num_allow = int(TARGET_LOG_ENTRIES * allow_ratio)
#     num_deny = TARGET_LOG_ENTRIES - num_allow

#     print(f"Generating exactly {num_allow} targeted 'Allow' logs...")
#     allow_count = 0
#     while allow_count < num_allow:
#         target_obj_name = random.choice(list(ACCESS_CONTROL_LISTS.keys()))
#         acl_for_obj = ACCESS_CONTROL_LISTS.get(target_obj_name, {})
#         if not acl_for_obj: continue

#         user_with_permission_name = random.choice(list(acl_for_obj.keys()))
#         user = next((u for u in all_users if u['user_name'] == user_with_permission_name), None)
#         target_obj = next((o for o in all_objects if o['object_name'] == target_obj_name), None)
        
#         if not user or not target_obj: continue

#         action = random.choice(list(acl_for_obj[user_with_permission_name]))
        
#         time_str = (start_time + timedelta(seconds=allow_count * 20)).strftime("%H:%M:%S")
#         generated_logs.append(f"<{user['user_name']} {target_obj['object_name']} {action} {time_str} Allow>")
#         allow_count += 1

#     print(f"Generating exactly {num_deny} targeted 'Deny' logs...")
#     deny_count = 0
#     while deny_count < num_deny:
#         user, obj = random.choice(all_users), random.choice(all_objects)
#         allowed_actions = ACCESS_CONTROL_LISTS.get(obj['object_name'], {}).get(user['user_name'], set())
#         denied_actions = [a for a in ACTIONS if a not in allowed_actions]
        
#         if denied_actions:
#             action = random.choice(denied_actions)
#             time_str = (start_time + timedelta(seconds=(num_allow + deny_count) * 20)).strftime("%H:%M:%S")
#             generated_logs.append(f"<{user['user_name']} {obj['object_name']} {action} {time_str} Deny>")
#             deny_count += 1

#     random.shuffle(generated_logs)

#     print(f"\nWriting {len(generated_logs)} logs to {LOG_FILE}.")
#     with open(LOG_FILE, "w") as f: f.write("\n".join(generated_logs))
        
#     decision_counts = collections.Counter(log.split()[-1].replace("'", "").replace('>', '') for log in generated_logs)
#     total_count = sum(decision_counts.values()) or 1
    
#     print("\n--- Generation Complete for DAC dataset! ---")
#     print(f"Final Log Ratio: {decision_counts['Allow']} Allow ({decision_counts['Allow']/total_count:.1%}) | {decision_counts['Deny']} Deny ({decision_counts['Deny']/total_count:.1%})")

# if __name__ == "__main__":
#     main()
import random
from datetime import datetime, timedelta
import collections

# --- Configuration ---
TARGET_LOG_ENTRIES = 2000
USER_FILE = "dac_user.txt"
OBJECT_FILE = "dac_object.txt"
LOG_FILE = "dac_logs.txt"

# --- 1. Define Data Structures ---

DEPARTMENTS = {
    "Finance": ["Analyst", "Senior_Analyst", "Manager", "Director"],
    "Sales": ["Representative", "Senior_Representative", "Manager", "Director"],
    "IT": ["Support_Specialist", "System_Admin", "Network_Engineer", "IT_Manager"],
    "HR": ["Generalist", "Recruiter", "HR_Manager", "Director"],
    "Engineering": ["Software_Engineer", "Senior_Engineer", "Lead_Engineer", "Engineering_Manager"],
    "Marketing": ["Coordinator", "Specialist", "Marketing_Manager", "Director"],
    "Legal": ["Paralegal", "Analyst", "General_Counsel", "Chief_Legal_Officer"],
    "Operations": ["Coordinator", "Operations_Manager", "Director_of_Operations"],
    "Customer_Support": ["Agent", "Team_Lead", "Support_Manager"],
    "Research": ["Researcher", "Senior_Researcher", "Lead_Scientist"]
}

OBJECTS_STRUCTURE = {
    "Financial": ["Low", "Medium", "High"], 
    "Operational": ["Low", "Medium"],
    "System": ["Medium", "High"], 
    "HR": ["Medium", "High", "Confidential"],
    "Source_Code": ["Medium", "High"], 
    "Legal": ["Medium", "High", "Confidential"],
    "Marketing": ["Low", "Medium"], 
    "Customer_Data": ["High", "Confidential"], 
    "General": ["Low"]
}

ACTIONS = ["read", "write", "execute", "delete"]

# Department ownership mapping
OWNING_DEPARTMENT = {
    "Financial": "Finance", "Operational": "Sales", "System": "IT",
    "HR": "HR", "Source_Code": "Engineering", "Legal": "Legal",
    "Marketing": "Marketing", "Customer_Data": "Customer_Support", "General": "HR",
}

# Cross-department business workflows
CROSS_DEPT_ACCESS = {
    "Finance": ["Operational", "Sales"],  # Finance needs sales data
    "Sales": ["Financial", "Marketing"],  # Sales needs financial reports and marketing
    "IT": ["System"],  # IT maintains systems
    "HR": ["HR", "General"],  # HR accesses HR and general docs
    "Legal": ["Legal", "HR"],  # Legal reviews HR contracts
    "Engineering": ["Source_Code", "System"],  # Engineers need code and systems
    "Operations": ["Operational", "Financial"],  # Ops needs operational and finance data
}

# Role hierarchy (higher roles inherit lower role permissions)
ROLE_HIERARCHY = {
    "Director": 4, "Chief_Legal_Officer": 4, "Engineering_Manager": 4, 
    "IT_Manager": 4, "HR_Manager": 4, "Marketing_Manager": 4,
    "Operations_Manager": 4, "Support_Manager": 4, "Lead_Scientist": 4,
    "Manager": 3, "Lead_Engineer": 3, "General_Counsel": 3,
    "Senior_Analyst": 2, "Senior_Representative": 2, "Senior_Engineer": 2,
    "Network_Engineer": 2, "System_Admin": 2, "Senior_Researcher": 2,
    "Analyst": 1, "Representative": 1, "Software_Engineer": 1,
    "Support_Specialist": 1, "Generalist": 1, "Recruiter": 1,
    "Coordinator": 1, "Paralegal": 1, "Agent": 1, "Researcher": 1,
}

ACCESS_CONTROL_LISTS = {}

# --- 2. The DAC Decision Logic ---
def get_decision(user, obj, action):
    """Check if user has permission for action on object."""
    user_permissions = ACCESS_CONTROL_LISTS.get(obj['object_name'], {}).get(user['user_name'], set())
    return "Allow" if action in user_permissions else "Deny"

# --- 3. Improved Data Generation with Realistic ACLs ---

def generate_users():
    """Generate users with department and designation."""
    users = []
    for dept, designations in DEPARTMENTS.items():
        for i in range(len(designations) * 2):
            user_name = f"{random.choice(['alex','casey','drew','jordan']).lower()}_{dept.lower()}_{i}"
            users.append({
                "user_name": user_name, 
                "department": dept, 
                "designation": random.choice(designations)
            })
    return users

def get_role_level(designation):
    """Get hierarchical level of a role."""
    return ROLE_HIERARCHY.get(designation, 1)

def generate_realistic_acls(all_users, obj_name, obj_type, obj_sens):
    """
    Generate realistic ACLs based on:
    - Sensitivity (controls ACL size)
    - Role hierarchy (managers get more permissions)
    - Business workflows (cross-department access)
    - Action realism (read > write > execute > delete)
    """
    acl = {}
    owner_dept = OWNING_DEPARTMENT.get(obj_type, "IT")
    
    # 1. Assign owner (senior role in owning department)
    dept_users = [u for u in all_users if u['department'] == owner_dept]
    if dept_users:
        # Prefer managers/directors as owners
        senior_users = [u for u in dept_users if get_role_level(u['designation']) >= 3]
        owner = random.choice(senior_users) if senior_users else random.choice(dept_users)
        
        # Owner gets full permissions based on role level
        if get_role_level(owner['designation']) >= 3:
            acl[owner['user_name']] = {'read', 'write', 'execute', 'delete'}
        else:
            acl[owner['user_name']] = {'read', 'write'}
    
    # 2. Add same-department collaborators (sensitivity-based sizing)
    if obj_sens == "Low":
        num_dept_collab = random.randint(5, 8)
    elif obj_sens == "Medium":
        num_dept_collab = random.randint(3, 5)
    elif obj_sens == "High":
        num_dept_collab = random.randint(1, 3)
    else:  # Confidential
        num_dept_collab = random.randint(0, 1)
    
    dept_pool = [u for u in dept_users if u['user_name'] not in acl]
    for _ in range(min(num_dept_collab, len(dept_pool))):
        collab = random.choice(dept_pool)
        dept_pool.remove(collab)
        
        role_level = get_role_level(collab['designation'])
        if role_level >= 3:  # Managers
            acl[collab['user_name']] = {'read', 'write'}
        elif role_level >= 2:  # Senior roles
            acl[collab['user_name']] = {'read', 'write'} if random.random() < 0.6 else {'read'}
        else:  # Junior roles
            acl[collab['user_name']] = {'read'}
    
    # 3. Add cross-department access (business workflows)
    if obj_sens in ["Low", "Medium"]:  # Only for non-sensitive data
        cross_depts = CROSS_DEPT_ACCESS.get(owner_dept, [])
        for related_type in cross_depts:
            if obj_type == related_type or random.random() < 0.4:  # 40% chance
                cross_users = [u for u in all_users 
                              if u['department'] != owner_dept 
                              and u['user_name'] not in acl
                              and (obj_type in CROSS_DEPT_ACCESS.get(u['department'], []))]
                
                if cross_users:
                    num_cross = random.randint(1, 3) if obj_sens == "Low" else random.randint(0, 2)
                    for _ in range(min(num_cross, len(cross_users))):
                        cross_user = random.choice(cross_users)
                        cross_users.remove(cross_user)
                        
                        # Cross-department users typically get read-only
                        if get_role_level(cross_user['designation']) >= 3:
                            acl[cross_user['user_name']] = {'read', 'write'} if random.random() < 0.3 else {'read'}
                        else:
                            acl[cross_user['user_name']] = {'read'}
    
    # 4. IT always gets read access to System objects
    if obj_type == "System":
        it_users = [u for u in all_users if u['department'] == "IT" and u['user_name'] not in acl]
        for it_user in random.sample(it_users, min(3, len(it_users))):
            acl[it_user['user_name']] = {'read', 'execute'}
    
    # 5. Legal gets read access to Legal and HR Confidential
    if obj_type in ["Legal", "HR"] and obj_sens in ["High", "Confidential"]:
        legal_users = [u for u in all_users 
                      if u['department'] == "Legal" 
                      and u['user_name'] not in acl
                      and get_role_level(u['designation']) >= 2]
        for legal_user in random.sample(legal_users, min(2, len(legal_users))):
            acl[legal_user['user_name']] = {'read'}
    
    return acl

def generate_objects_and_acls(all_users):
    """Generate objects with realistic, business-driven ACLs."""
    print("Generating objects and realistic, role/sensitivity-based ACLs...")
    objects = []
    
    for o_type, sensitivities in OBJECTS_STRUCTURE.items():
        for sens in sensitivities:
            for i in range(3):
                object_name = f"{o_type.lower()}_{sens.lower()}_{i}.dat"
                objects.append({
                    "object_name": object_name, 
                    "type": o_type, 
                    "sensitivity": sens
                })
                
                # Generate realistic ACL
                acl = generate_realistic_acls(all_users, object_name, o_type, sens)
                ACCESS_CONTROL_LISTS[object_name] = acl
    
    # Print ACL statistics
    acl_sizes = [len(acl) for acl in ACCESS_CONTROL_LISTS.values()]
    print(f"ACL Statistics: Min={min(acl_sizes)}, Max={max(acl_sizes)}, Avg={sum(acl_sizes)/len(acl_sizes):.1f}")
    
    return objects

def format_for_file(entity, is_user=True):
    """Format entity for file output."""
    if is_user:
        return f"<{entity['user_name']} department:{entity['department']} designation:{entity['designation']}>"
    else:
        return f"<{entity['object_name']} type:{entity['type']} sensitivity:{entity['sensitivity']}>"

# --- 4. Main Script with Dynamic Allow/Deny Ratio ---

def main():
    print("--- Starting Improved DAC Log Generation ---")
    
    all_users = generate_users()
    all_objects = generate_objects_and_acls(all_users)
    
    with open(USER_FILE, "w") as f:
        f.write("\n".join([format_for_file(u) for u in all_users]))
    with open(OBJECT_FILE, "w") as f:
        f.write("\n".join([format_for_file(o, is_user=False) for o in all_objects]))
    
    generated_logs = []
    start_time = datetime.now() - timedelta(hours=12)
    
    # Dynamic ratio: Generate more realistic distribution
    # Real DAC: users mostly access what they're allowed to, but denials happen
    allow_ratio = 0.90  # 70% Allow, 30% Deny (more realistic than 90%)
    num_allow = int(TARGET_LOG_ENTRIES * allow_ratio)
    num_deny = TARGET_LOG_ENTRIES - num_allow
    
    print(f"Generating {num_allow} 'Allow' logs...")
    allow_count = 0
    attempts = 0
    while allow_count < num_allow and attempts < num_allow * 10:
        attempts += 1
        obj_name = random.choice(list(ACCESS_CONTROL_LISTS.keys()))
        acl = ACCESS_CONTROL_LISTS.get(obj_name, {})
        if not acl:
            continue
        
        user_name = random.choice(list(acl.keys()))
        user = next((u for u in all_users if u['user_name'] == user_name), None)
        obj = next((o for o in all_objects if o['object_name'] == obj_name), None)
        
        if not user or not obj:
            continue
        
        # Realistic action distribution: read >> write > execute > delete
        action_weights = {'read': 0.6, 'write': 0.25, 'execute': 0.10, 'delete': 0.05}
        available_actions = list(acl[user_name])
        if available_actions:
            # Weight by action frequency
            weighted_actions = [a for a in available_actions for _ in range(int(action_weights.get(a, 0.1) * 100))]
            action = random.choice(weighted_actions) if weighted_actions else random.choice(available_actions)
            
            time_str = (start_time + timedelta(seconds=allow_count * 15)).strftime("%H:%M:%S")
            generated_logs.append(f"<{user['user_name']} {obj['object_name']} {action} {time_str} Allow>")
            allow_count += 1
    
    print(f"Generating {num_deny} 'Deny' logs...")
    deny_count = 0
    attempts = 0
    while deny_count < num_deny and attempts < num_deny * 10:
        attempts += 1
        user = random.choice(all_users)
        obj = random.choice(all_objects)
        
        allowed_actions = ACCESS_CONTROL_LISTS.get(obj['object_name'], {}).get(user['user_name'], set())
        denied_actions = [a for a in ACTIONS if a not in allowed_actions]
        
        if denied_actions:
            # Weight denials: more read denials than delete denials
            action_weights = {'read': 0.5, 'write': 0.3, 'execute': 0.15, 'delete': 0.05}
            weighted_actions = [a for a in denied_actions for _ in range(int(action_weights.get(a, 0.1) * 100))]
            action = random.choice(weighted_actions) if weighted_actions else random.choice(denied_actions)
            
            time_str = (start_time + timedelta(seconds=(num_allow + deny_count) * 15)).strftime("%H:%M:%S")
            generated_logs.append(f"<{user['user_name']} {obj['object_name']} {action} {time_str} Deny>")
            deny_count += 1
    
    random.shuffle(generated_logs)
    
    print(f"\nWriting {len(generated_logs)} logs to {LOG_FILE}.")
    with open(LOG_FILE, "w") as f:
        f.write("\n".join(generated_logs))
    
    decision_counts = collections.Counter(log.split()[-1].rstrip('>') for log in generated_logs)
    total_count = sum(decision_counts.values()) or 1
    
    print("\n--- DAC Generation Complete! ---")
    print(f"Users: {len(all_users)}, Objects: {len(all_objects)}")
    print(f"Final Log Ratio: {decision_counts['Allow']} Allow ({decision_counts['Allow']/total_count:.1%}) | "
          f"{decision_counts['Deny']} Deny ({decision_counts['Deny']/total_count:.1%})")

if __name__ == "__main__":
    main()
