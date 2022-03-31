function y=frob_norm_3D(X)
    %Frobeniuns norm of a third-order tensor
    A=X.^2;
    y=sqrt(sum(sum(sum(A))));
end