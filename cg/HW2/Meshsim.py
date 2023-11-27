from utils.mesh import Mesh

def main():
    mesh = Mesh("Datas/Buddha.obj")
    v1 = int(len(mesh.vs) * 0.01) #0.5 0.2 0.01
    simp = mesh.simplification(target_v=v1, valence_aware=False)
    simp.save("Results/Buddha_{}.obj".format(len(simp.vs)))
    print("The simplified obj have been saved in results.")

if __name__ == "__main__":
    main()