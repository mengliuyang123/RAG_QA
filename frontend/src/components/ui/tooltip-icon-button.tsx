import { Slot } from "radix-ui";
import { forwardRef, type ComponentPropsWithRef } from "react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

// 改写自 assistant-ui 官方 registry 的 tooltip-icon-button。
export type TooltipIconButtonProps = ComponentPropsWithRef<typeof Button> & {
  tooltip: string;
  side?: "top" | "bottom" | "left" | "right";
};

export const TooltipIconButton = forwardRef<HTMLButtonElement, TooltipIconButtonProps>(
  ({ children, tooltip, side = "bottom", className, ...rest }, ref) => (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          variant="ghost"
          size="icon"
          {...rest}
          className={cn("size-6 p-1 active:scale-90", className)}
          ref={ref}
        >
          <Slot.Slottable>{children}</Slot.Slottable>
          <span className="sr-only">{tooltip}</span>
        </Button>
      </TooltipTrigger>
      <TooltipContent side={side}>{tooltip}</TooltipContent>
    </Tooltip>
  ),
);
TooltipIconButton.displayName = "TooltipIconButton";
