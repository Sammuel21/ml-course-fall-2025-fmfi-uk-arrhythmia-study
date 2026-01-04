% NOTE (README.md): GitHub READMEs do not render full LaTeX documents (e.g., \section, \subsection).
% You can paste this as a fenced code block, or move it into a .tex report. Math ($...$) will render.

\section{Why AP micro/macro signal a better model}

\paragraph{What Average Precision (AP) measures.}
Average Precision (AP) summarizes the precision--recall curve \emph{across all possible thresholds}. In other words, it evaluates the \textbf{ranking quality} of the model's scores without committing to any single decision cutoff. If AP increases, the model is better at placing true positives above false positives in its score ordering, independent of threshold choice.

\subsection{AP micro: the best single global optimization indicator (multilabel)}
With micro-averaging, evaluation aggregates decisions over all \emph{(sample, label)} pairs into one pool. This makes \textbf{AP\_micro} a strong global indicator of improvement for multilabel problems:
\begin{itemize}
  \item Interpretation: ``Across all label decisions, how well does the model rank true positives ahead of false positives?''
  \item Why it fits imbalanced multilabel: precision--recall based metrics remain informative when positives are rare and the label space is sparse.
\end{itemize}

\subsection{AP macro: sensitivity to rare labels}
With macro-averaging, AP is computed per label and then averaged equally across labels:
\begin{itemize}
  \item Interpretation: ``How good is ranking quality for the \emph{average label}, treating rare and common labels equally?''
  \item Practical use: AP\_macro provides evidence that improvements extend beyond frequent labels into the long tail of rare conditions.
\end{itemize}

\section{What additional value F1 micro/macro provides after threshold tuning}

\paragraph{Why F1 is different from AP.}
AP answers: \textbf{``Are the scores good?''} (threshold-free).
F1 answers: \textbf{``If we must output 0/1 labels, how good are the decisions at a chosen operating point?''}
F1 depends on hard predictions, so it changes when you change thresholds even if the underlying ranking quality (AP) is unchanged.

\subsection{Why reporting F1 after threshold tuning is useful}
A threshold of $0.5$ is not inherently meaningful under heavy class imbalance. Threshold tuning (on validation data) converts good ranking into useful discrete predictions:
\begin{itemize}
  \item A model may have high AP but low F1 at threshold $0.5$ simply because the cutoff is poorly chosen.
  \item After tuning thresholds on a validation split, the reported F1 reflects a \textbf{justified decision policy} derived from the model's scores.
  \item This gives readers an intuitive ``final label quality'' number, complementary to AP.
\end{itemize}

\subsection{Interpreting micro vs macro F1}
\begin{itemize}
  \item \textbf{F1\_micro}: aggregates TP/FP/FN globally across all labels.
    \begin{itemize}
      \item Interpretation: ``Overall, how good are our final binary decisions across all label assignments?''
      \item Practical meaning: dominated by frequent labels, but still accounts for all decisions.
    \end{itemize}
  \item \textbf{F1\_macro}: average of per-label F1 values (each label counts equally).
    \begin{itemize}
      \item Interpretation: ``How well do we perform for the average label, including rare ones?''
      \item Practical meaning: can be low/noisy with extremely rare labels, but useful to see tail improvements.
    \end{itemize}
\end{itemize}

\section{Recommended reporting structure}

\subsection{Model comparison / optimization tracking (threshold-free)}
Use these to decide whether one model configuration is intrinsically better:
\begin{itemize}
  \item Primary: \textbf{AP\_micro}
  \item Secondary: \textbf{AP\_macro} (rare-label sensitivity)
\end{itemize}

\subsection{Final operating point (thresholded, tuned on validation only)}
After selecting the best estimator using threshold-free metrics:
\begin{itemize}
  \item Primary: \textbf{F1\_micro} at tuned threshold(s)
  \item Secondary: \textbf{F1\_macro} at tuned threshold(s)
  \item Optional but recommended: micro precision and micro recall at the tuned threshold(s) to make the trade-off explicit.
\end{itemize}

\paragraph{Reader takeaway.}
\begin{itemize}
  \item \textbf{AP improves} $\Rightarrow$ the model's scoring/ranking is genuinely better (not a threshold artifact).
  \item \textbf{F1 improves after threshold tuning} $\Rightarrow$ those better scores can be converted into better discrete label predictions at a justified operating point.
\end{itemize}
