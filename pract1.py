# ----------------------------
# Part 1: Set Operations
# ----------------------------
A = set(map(int, input("Enter elements of set A: ").split()))
B = set(map(int, input("Enter elements of set B: ").split()))

print("Union:", A | B)
print("Intersection:", A & B)
print("Difference (A - B):", A - B)
print("Cartesian Product:", {(a, b) for a in A for b in B})


# ----------------------------
# Part 2: Relation Properties
# ----------------------------
n = int(input("Enter number of ordered pairs in relation R: "))
R = set()

print("Enter pairs (a b):")
for _ in range(n):
    a, b = map(int, input().split())
    R.add((a, b))

# Reflexive
reflexive = all((a, a) in R for a in A)

# Symmetric
symmetric = all((b, a) in R for (a, b) in R)

# Transitive
transitive = True
for (a, b) in R:
    for (c, d) in R:
        if b == c and (a, d) not in R:
            transitive = False
            break

print("Reflexive:", reflexive)
print("Symmetric:", symmetric)
print("Transitive:", transitive)


# ----------------------------
# Part 3: Function Properties
# ----------------------------
m = int(input("Enter number of function mappings f: "))
f = {}

print("Enter mappings (a b) meaning a -> b:")
for _ in range(m):
    a, b = map(int, input().split())
    f[a] = b

# Injective
injective = len(set(f.values())) == len(f.values())

# Surjective (onto B)
surjective = set(f.values()) == B

# Bijective
bijective = injective and surjective

print("Injective:", injective)
print("Surjective:", surjective)
print("Bijective:", bijective)
